# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# try to manage it to adapt to deepspeed framework with extra concept involved

from __future__ import absolute_import, division, print_function
import argparse
import os
import base64
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed, weighted_sample
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification, test_clip, ImageBertForSequenceClassification2
from transformers.pytorch_transformers import BertTokenizer, BertConfig 
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from oscar.sequential_eval_utils import *


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True, reranking_mode='i2t', img_feat=None):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),
        reranking_mode: 'i2t' or 't2i', used for re-ranking setting!

        """
        super(RetrievalDataset, self).__init__()
        assert reranking_mode in ['i2t', 't2i', 'left']
        self.re_mode = reranking_mode
        self.img_file = args.img_feat_file
        caption_file = op.join(args.data_dir, '{}_captions.pt'.format(split))
        # self.img_tsv = TSVFile(self.img_file)
        self.split = split
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        self.num_of_total_captions = sum([len(self.captions[k]) for k in self.img_keys])

        # get the image image_id to index map
        # imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        # self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
        # get the image features and labels
        print(1)
        if img_feat is not None:
            self.img_feats = img_feat
        else:
            img_feat_file = op.join(args.data_dir, '{}_img_{}_feats.pt'.format(split, args.img_feature_type))
            self.img_feats = torch.load(img_feat_file)

        # get extra concepts
        print(2)
        if args.extra_concept:
            add_concept_file = op.join(args.data_dir, '{}_extra_concepts.pt'.format(split))
            self.extra_concep = torch.load(add_concept_file)
            if args.concept2id_file:
                concept2id = torch.load(op.join(args.data_dir, args.concept2id_file))
                for k,v in self.extra_concep.items():
                    # pre-process to leave only 1 word concepts!
                    filtered = []
                    for ex_con in v:
                        single_concept = [concept2id[c] if c in concept2id else concept2id['[UNK]'] for c in ex_con]
                        filtered.append(single_concept)
                    self.extra_concep[k] = filtered
                del(concept2id)
        
        print(3)
        if args.add_od_labels:
            labels_file = op.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))
            self.labels = torch.load(labels_file)

        print(4)
        if args.clip_neg_sampling and is_train:
            neg_scpres_file = op.join(args.data_dir, '{}_clip_ft_scores.pt'.format(split))
            if args.try_reranking:
                neg_scpres_file = op.join(args.data_dir, '{}_pre_ranking2.pt'.format(split))
            self.neg_scores = torch.load(neg_scpres_file)

        print(5)
        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
            if self.num_captions_per_img < 5 and args.less_cap_train:
                # providing only less captions for one image
                if op.exists(op.join(args.data_dir, '{}_cap_permute.pt'.format(split))):
                    self.sub_cap_index = torch.load(op.join(args.data_dir, '{}_cap_permute.pt'.format(split)))
                else:
                    self.sub_cap_index = {}
                    for img in self.img_keys:
                        cap_idx = [i for i in range(5)]
                        random.shuffle(cap_idx)
                        self.sub_cap_index[img] = cap_idx
                    torch.save(self.sub_cap_index, op.join(args.data_dir, '{}_cap_permute.pt'.format(split)))


        else:
            if args.less_cap_train and split=='train':
                self.num_captions_per_img = 5 - args.num_captions_per_img_train
                self.num_of_total_captions = len(self.img_keys) * self.num_captions_per_img
                self.sub_cap_index = torch.load(op.join(args.data_dir, '{}_cap_permute.pt'.format(split)))
            else:
                self.num_captions_per_img = args.num_captions_per_img_val
                self.num_images_per_cap = args.num_images_per_cap_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}

            if args.eval_caption_index_file and reranking_mode=='i2t':
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indexs = True
                assert not args.cross_image_eval 
                if split == 'train':
                    caption_index_file = op.join(args.data_dir, args.train_sub_index)
                    self.caption_indexs = torch.load(caption_index_file)
                    self.caption_indexs = {k:v for k,v in self.caption_indexs.items() if isinstance(k, int)}
                else:
                    caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                    self.caption_indexs = torch.load(caption_index_file)
                if not type(self.caption_indexs[self.img_keys[0]]) == list:
                    self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
                self.has_image_indexs = False
            else:
                self.has_caption_indexs = False
            
            if args.eval_image_index_file and reranking_mode=='t2i':
                # hard negative image/caption indexs for retrieval re-rank setting.
                # useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_image_indexs = True
                assert not args.cross_image_eval 
                if split == 'train':
                    image_index_file = op.join(args.data_dir, args.train_sub_index)
                    self.image_indexs = torch.load(image_index_file)
                    self.image_indexs = {k:v for k,v in self.image_indexs.items() if isinstance(k, tuple)}
                else:
                    image_index_file = op.join(args.data_dir, args.eval_image_index_file)
                    self.image_indexs = torch.load(image_index_file)
                    caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                    self.caption_indexs = torch.load(caption_index_file)
                self.has_caption_indexs = False
            else:
                self.has_image_indexs = False

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.less_cap_train and self.re_mode=='left':
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1 = self.img_keys[img_idx]
            return img_idx, [img_key1, self.sub_cap_index[img_key1][cap_idx+self.args.num_captions_per_img_train]]
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            if self.split == 'train':
                img_idx = index // self.args.num_captions_per_img_val
                cap_idx = index % self.args.num_captions_per_img_val
            else:
                img_idx = index // self.num_captions_per_img
                cap_idx = index % self.num_captions_per_img
            if self.args.try_reranking:
                (img_key1, cap_idx1), pre_score = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
                return img_idx, [img_key1, cap_idx1], pre_score
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            if isinstance(img_key1, tuple):
                img_key1, cap_idx1 = img_key1
            return img_idx, [img_key1, cap_idx1]
        if not self.is_train and self.has_image_indexs:
            if self.split == 'train':
                cap_idx = index // self.args.num_images_per_cap_val
                cap_img_idx = cap_idx // self.num_captions_per_img
                cap_cap_idx = cap_idx % self.num_captions_per_img
                cap_cap_idx = self.sub_cap_index[self.img_keys[cap_img_idx]][cap_cap_idx+self.args.num_captions_per_img_train]
                img_idx = index % self.args.num_images_per_cap_val
            else:
                cap_idx = index // self.num_images_per_cap
                cap_img_idx = cap_idx // 5
                cap_cap_idx = cap_idx % 5
                img_idx = index % self.num_images_per_cap
            if self.args.try_reranking:
                img_key1, pre_score = self.image_indexs[(self.img_keys[cap_img_idx],cap_cap_idx)][img_idx]
                return img_key1, [self.img_keys[cap_img_idx], cap_cap_idx], pre_score
            img_key1 = self.image_indexs[(self.img_keys[cap_img_idx],cap_cap_idx)][img_idx]
            if isinstance(img_key1, tuple) or isinstance(img_key1, list):
                img_key1, pr_score = img_key1
            return img_key1, [self.img_keys[cap_img_idx], cap_cap_idx]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        if self.num_captions_per_img < 5:
            cap_idx = self.sub_cap_index[self.img_keys[img_idx]][cap_idx]
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        if self.args.try_reranking:
            img_idx, cap_idx, pr_score = self.get_image_caption_index(index)
        else:
            img_idx, cap_idx = self.get_image_caption_index(index)
        if img_idx in self.img_keys:
            return 1 if img_idx == cap_idx[0] else 0
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            
            # if cap_index is not None:
            #     extra_concepts = self.extra_concep[str(img_key)][cap_index]
            #     if self.args.num_extra_concept > 0 and self.args.num_extra_concept < len(extra_concepts):
            #         # extra_concepts = random.sample(extra_concepts, self.args.num_extra_concept)
            #         extra_concepts = extra_concepts[:self.args.num_extra_concept]
            #     od_labels += ' '.join(od_labels)
            return od_labels

    def get_extra_concepts(self, img_key, cap_index):
        if not self.args.extra_concept:
            return ''
        extra_concepts = self.extra_concep[str(img_key)][cap_index]
        if self.args.num_extra_concept > 0 and self.args.num_extra_concept < len(extra_concepts):
            # extra_concepts = random.sample(extra_concepts, self.args.num_extra_concept)
            extra_concepts = extra_concepts[:self.args.num_extra_concept]
        if len(extra_concepts) == 0:
            return []
        if type(extra_concepts[0]) == str:
            return ' '.join(extra_concepts)
        else:
            return extra_concepts

    def get_image_related(self, img_key, cap_id=None):
        # get all information related to this image and its 4 captions
        if self.split != 'train':
            pse_bg = self.caption_indexs[img_key][:self.args.num_captions_per_img_train]
            all_caps = []
            all_concepts = []
            for i in pse_bg:
                cap_index = i[0]
                all_caps.append(self.captions[cap_index[0]][cap_index[1]])
                all_concepts += self.extra_concep[str(cap_index[0])][cap_index[1]]
            all_caps = self.tokenizer.sep_token.join(all_caps)
            return all_caps, all_concepts
        noise_include = self.args.include_noise
        all_caps = []
        all_concepts = []
        if cap_id == -1:
            cap_id = random.choice(self.sub_cap_index[img_key][:self.args.num_captions_per_img_train])
        c = 0
        for i in self.sub_cap_index[img_key][:self.args.num_captions_per_img_train]:
            if cap_id is not None:
                if i == cap_id:
                    continue
                if cap_id == -2:
                    if c == self.args.num_captions_per_img_train - 1:
                        break
            all_caps.append(self.captions[img_key][i])
            all_concepts += self.extra_concep[str(img_key)][i]
            c += 1
        assert(c==(self.args.num_captions_per_img_train-1) or c==self.args.num_captions_per_img_train)
        con_position = None
        con_label = None
        if self.is_train and noise_include:
            neg_cap = random.sample(self.neg_scores[img_key], 1)[0]
            if not isinstance(neg_cap[0], int):
                neg_cap = neg_cap[0]
            all_caps.append(self.captions[neg_cap[0]][neg_cap[1]])
            random.shuffle(all_caps)
            if self.args.neg_con_detect>0:
                pos_con = set(all_concepts)
                neg_con = set(self.extra_concep[str(neg_cap[0])][neg_cap[1]])-pos_con
                con2pol = {}
                for pc in pos_con:
                    con2pol[pc] = 1
                for nc in neg_con:
                    con2pol[nc] = -1
                all_concepts = list(set.union(pos_con, neg_con))
                random.shuffle(all_concepts)
                all_concepts = all_concepts[:self.args.bg_con_num]
                pos_pool = []
                neg_pool = []
                for c_i, c_c in enumerate(all_concepts):
                    if con2pol[c_c] == 1:
                        pos_pool.append(c_i)
                    else:
                        neg_pool.append(c_i)
                if len(neg_pool) == 0:
                    pos_p = 1
                else:
                    pos_p = 0.5
                if len(pos_pool) == 0:
                    pos_p = -1
                    if len(neg_pool)==0:
                        for c_i in range(self.args.neg_con_detect):
                            sample_con = random.randint(1, self.args.concept_size)
                            sample_con = self.args.vocab_size + sample_con - 1
                            all_concepts.append(sample_con)
                            neg_pool.append(c_i)
                start_pos = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len
                con_label = []
                con_position = []
                for _ in range(self.args.neg_con_detect):
                    if random.random() <= pos_p:
                        # sample a positive con
                        con_label.append(1)
                        con_position.append(start_pos+random.sample(pos_pool, 1)[0])
                    else:
                        # sample a negative con
                        con_label.append(0)
                        con_position.append(start_pos+random.sample(neg_pool, 1)[0])
            # all_concepts += self.extra_concep[str(neg_cap[0])][neg_cap[1]]
            # random.shuffle(all_concepts)
        all_caps = self.tokenizer.sep_token.join(all_caps)
        return all_caps, all_concepts, con_position, con_label


    def tensorize_example2(self, text_a, img_feat, text_b=None, text_c=None,
            ex_text=None, ex_con=None, cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_c_segment_id=0):
        tokens_a = self.tokenizer.tokenize(text_a)
        if text_b:
            num_extra_tokens = 3
        else:
            num_extra_tokens = 2
        if len(tokens_a) > self.args.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.args.max_seq_length - num_extra_tokens)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                num_left_tokens = max(0, self.max_seq_len - len(tokens) - 1) # to avoid -1
                assert(num_left_tokens >= 0)
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            additional_segment_ids = [sequence_b_segment_id] * (len(tokens_b) + 1)
            seq_b_len = len(tokens_b) + 1
            segment_ids += additional_segment_ids

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if text_c is not None:
            if len(text_c) > self.args.num_extra_concept:
                text_c = text_c[:self.args.num_extra_concept]
            concept_len = len(text_c)
            con_pad_len = self.args.num_extra_concept - concept_len
            text_c += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * con_pad_len) # to avoid type error
            segment_ids += [sequence_c_segment_id]*concept_len + [pad_token_segment_id]*con_pad_len
            input_ids += text_c
        else:
            concept_len = 0
            con_pad_len = 0

        if ex_text is not None:
            ex_tokens = self.tokenizer.tokenize(ex_text)
            if len(ex_tokens) > self.args.bg_seq_len:
                ex_tokens = ex_tokens[:self.args.bg_seq_len]
            ex_text_len = len(ex_tokens)
            ex_text_pad_len = self.args.bg_seq_len - ex_text_len
            ex_tokens += [self.tokenizer.pad_token] * ex_text_pad_len
            input_ids += self.tokenizer.convert_tokens_to_ids(ex_tokens)
            segment_ids += [sequence_b_segment_id] * self.args.bg_seq_len

        if ex_con is not None:
            if len(ex_con) > self.args.bg_con_num:
                ex_con = ex_con[:self.args.bg_con_num]
            ex_con_len = len(ex_con)
            ex_con_pad_len = self.args.bg_con_num - ex_con_len
            ex_con += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token]*ex_con_pad_len)
            segment_ids += [sequence_b_segment_id] * self.args.bg_con_num
            input_ids += ex_con

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * concept_len + [0] * con_pad_len + \
                             [1] * ex_text_len + [0] * ex_text_pad_len + \
                             [1] * ex_con_len + [0] * ex_con_pad_len + \
                             [1] * img_len + [0] * img_padding_len
        elif att_mask_type == 'SG':
            # SG is short for Sub-Graph
            attention_mask = []
            max_len = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                      + self.args.bg_con_num + self.max_img_seq_len
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            # cc = caption concept
            cc_start, cc_end = self.max_seq_len, self.max_seq_len + concept_len
            # bc = background captions
            bc_start = self.max_seq_len + self.args.num_extra_concept
            bc_end = self.max_seq_len + self.args.num_extra_concept + ex_text_len
            # be = background extra concepts
            be_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len
            be_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + ex_con_len
            r_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + self.args.bg_con_num
            r_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                     + self.args.bg_con_num + img_len
            atm = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention inside a block
            atm[c_start : c_end, c_start : c_end] = 1
            atm[l_start : l_end, l_start : l_end] = 1
            atm[cc_start : cc_end, cc_start : cc_end] = 1
            atm[bc_start : bc_end, bc_start : bc_end] = 1
            atm[be_start : be_end, be_start : be_end] = 1
            atm[r_start : r_end, r_start : r_end] = 1
            # full attention between captions and concepts
            atm[c_start : c_end, cc_start : cc_end] = 1
            atm[cc_start : cc_end, c_start : c_end] = 1
            atm[bc_start : bc_end, be_start : be_end] = 1
            atm[be_start : be_end, bc_start : bc_end] = 1
            # full attention between background concepts/captions and images
            atm[bc_start:bc_end, r_start:r_end] = 1
            atm[r_start:r_end, bc_start:bc_end] = 1
            atm[be_start:be_end, r_start:r_end] = 1
            atm[r_start:r_end, be_start:be_end] = 1
            # full attention between labels and concept/images
            atm[l_start:l_end, r_start:r_end] = 1
            atm[r_start:r_end, l_start:l_end] = 1
            atm[l_start:l_end, be_start:be_end] = 1
            atm[be_start:be_end, l_start:l_end] = 1
            atm[l_start:l_end, bc_start:bc_end] = 1
            atm[bc_start:bc_end, l_start:l_end] = 1
            atm[l_start:l_end, cc_start:cc_end] = 1
            atm[cc_start:cc_end, l_start:l_end] = 1
            atm2 = atm.clone()
            atm2[cc_start:cc_end, r_start:r_end] = 1
            atm2[r_start:r_end, cc_start:cc_end] = 1
            # caption 2 images
            atm2[c_start:c_end, r_start:r_end] = 1
            atm2[r_start:r_end, c_start:c_end] = 1
            # between concepts
            atm2[cc_start:cc_end, be_start:be_end] = 1
            atm2[be_start:be_end, cc_start:cc_end] = 1
            # add attention between images and captions/cc
            atm2[c_start:c_end, l_start:l_end] = 1
            atm2[l_start:l_end, c_start:c_end] = 1
            atm2[cc_start:cc_end, r_start:r_end] = 1
            atm2[r_start:r_end, cc_start:cc_end] = 1
            # add attention between captions and background concepts
            atm2[(c_start+1):c_end, be_start:be_end] = 1
            atm2[be_start:be_end, (c_start+1):c_end] = 1
            # attention_mask.append(atm)
            attention_mask = (atm, atm2)
            
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        except TypeError:
            print(input_ids)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def tensorize_example(self, text_a, img_feat, text_b=None, text_c=None,
            ex_text=None, ex_con=None, cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_c_segment_id=0):
        tokens_a = self.tokenizer.tokenize(text_a)
        if text_b:
            num_extra_tokens = 3
        else:
            num_extra_tokens = 2
        if len(tokens_a) > self.args.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.args.max_seq_length - num_extra_tokens)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                num_left_tokens = max(0, self.max_seq_len - len(tokens) - 1) # to avoid -1
                assert(num_left_tokens >= 0)
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            additional_segment_ids = [sequence_b_segment_id] * (len(tokens_b) + 1)
            seq_b_len = len(tokens_b) + 1
            segment_ids += additional_segment_ids

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if text_c is not None:
            if len(text_c) > self.args.num_extra_concept:
                text_c = text_c[:self.args.num_extra_concept]
            concept_len = len(text_c)
            con_pad_len = self.args.num_extra_concept - concept_len
            text_c += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * con_pad_len) # to avoid type error
            segment_ids += [sequence_c_segment_id]*concept_len + [pad_token_segment_id]*con_pad_len
            input_ids += text_c
        else:
            concept_len = 0
            con_pad_len = 0

        if ex_text is not None:
            ex_tokens = self.tokenizer.tokenize(ex_text)
            if len(ex_tokens) > self.args.bg_seq_len:
                ex_tokens = ex_tokens[:self.args.bg_seq_len]
            ex_text_len = len(ex_tokens)
            ex_text_pad_len = self.args.bg_seq_len - ex_text_len
            ex_tokens += [self.tokenizer.pad_token] * ex_text_pad_len
            input_ids += self.tokenizer.convert_tokens_to_ids(ex_tokens)
            segment_ids += [sequence_b_segment_id] * self.args.bg_seq_len

        if ex_con is not None:
            if len(ex_con) > self.args.bg_con_num:
                ex_con = ex_con[:self.args.bg_con_num]
            ex_con_len = len(ex_con)
            ex_con_pad_len = self.args.bg_con_num - ex_con_len
            ex_con += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token]*ex_con_pad_len)
            segment_ids += [sequence_b_segment_id] * self.args.bg_con_num
            input_ids += ex_con
            # find common concepts

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * concept_len + [0] * con_pad_len + \
                             [1] * ex_text_len + [0] * ex_text_pad_len + \
                             [1] * ex_con_len + [0] * ex_con_pad_len + \
                             [1] * img_len + [0] * img_padding_len
        elif att_mask_type == 'SG':
            # SG is short for Sub-Graph
            attention_mask = []
            max_len = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                      + self.args.bg_con_num + self.max_img_seq_len
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            # cc = caption concept
            cc_start, cc_end = self.max_seq_len, self.max_seq_len + concept_len
            # bc = background captions
            bc_start = self.max_seq_len + self.args.num_extra_concept
            bc_end = self.max_seq_len + self.args.num_extra_concept + ex_text_len
            # be = background extra concepts
            be_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len
            be_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + ex_con_len
            r_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + self.args.bg_con_num
            r_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                     + self.args.bg_con_num + img_len
            atm = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention inside a block
            atm[c_start : c_end, c_start : c_end] = 1
            atm[l_start : l_end, l_start : l_end] = 1
            atm[cc_start : cc_end, cc_start : cc_end] = 1
            atm[bc_start : bc_end, bc_start : bc_end] = 1
            atm[be_start : be_end, be_start : be_end] = 1
            atm[r_start : r_end, r_start : r_end] = 1
            # full attention between captions and concepts
            atm[c_start : c_end, cc_start : cc_end] = 1
            atm[cc_start : cc_end, c_start : c_end] = 1
            atm[bc_start : bc_end, be_start : be_end] = 1
            atm[be_start : be_end, bc_start : bc_end] = 1
            # full attention between background concepts/captions and images
            atm[bc_start:bc_end, r_start:r_end] = 1
            atm[r_start:r_end, bc_start:bc_end] = 1
            atm[be_start:be_end, r_start:r_end] = 1
            atm[r_start:r_end, be_start:be_end] = 1
            # full attention between labels and concept/images
            atm[l_start:l_end, r_start:r_end] = 1
            atm[r_start:r_end, l_start:l_end] = 1
            atm[l_start:l_end, be_start:be_end] = 1
            atm[be_start:be_end, l_start:l_end] = 1
            atm[l_start:l_end, bc_start:bc_end] = 1
            atm[bc_start:bc_end, l_start:l_end] = 1
            atm2 = atm.clone()
            atm2[l_start:l_end, cc_start:cc_end] = 1
            atm2[cc_start:cc_end, l_start:l_end] = 1
            # caption 2 images
            atm2[c_start:c_end, r_start:r_end] = 1
            atm2[r_start:r_end, c_start:c_end] = 1
            # between concepts
            atm2[cc_start:cc_end, be_start:be_end] = 1
            atm2[be_start:be_end, cc_start:cc_end] = 1
            # add attention between images and captions/cc
            atm2[c_start:c_end, l_start:l_end] = 1
            atm2[l_start:l_end, c_start:c_end] = 1
            # add attention between captions and background concepts
            atm2[(c_start+1):c_end, be_start:be_end] = 1
            atm2[be_start:be_end, (c_start+1):c_end] = 1
            # attention_mask.append(atm)
            attention_mask = (atm, atm2)
            
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        except TypeError:
            print(input_ids)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def tensorize_example3(self, text_a, img_feat, text_b=None, text_c=None,
            ex_text=None, ex_con=None, cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_c_segment_id=0):
        tokens_a = self.tokenizer.tokenize(text_a)
        if text_b:
            num_extra_tokens = 3
        else:
            num_extra_tokens = 2
        if len(tokens_a) > self.args.max_seq_length - num_extra_tokens: # edited here to make it for sequence length == 68
            tokens_a = tokens_a[:(self.args.max_seq_length - num_extra_tokens)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                num_left_tokens = max(0, self.max_seq_len - len(tokens) - 1) # to avoid -1
                assert(num_left_tokens >= 0)
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            additional_segment_ids = [sequence_b_segment_id] * (len(tokens_b) + 1)
            seq_b_len = len(tokens_b) + 1
            segment_ids += additional_segment_ids

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # if ex_con is not None:
        #     ex_con = list(set.intersection(set(text_c), set(ex_con)))

        if text_c is not None:
            if len(text_c) > self.args.num_extra_concept:
                text_c = text_c[:self.args.num_extra_concept]
            concept_len = len(text_c)
            con_pad_len = self.args.num_extra_concept - concept_len
            text_c += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * con_pad_len) # to avoid type error
            segment_ids += [sequence_c_segment_id]*concept_len + [pad_token_segment_id]*con_pad_len
            input_ids += text_c
        else:
            concept_len = 0
            con_pad_len = 0

        if ex_text is not None:
            ex_tokens = self.tokenizer.tokenize(ex_text)
            if len(ex_tokens) > self.args.bg_seq_len:
                ex_tokens = ex_tokens[:self.args.bg_seq_len]
            ex_text_len = len(ex_tokens)
            ex_text_pad_len = self.args.bg_seq_len - ex_text_len
            ex_tokens += [self.tokenizer.pad_token] * ex_text_pad_len
            input_ids += self.tokenizer.convert_tokens_to_ids(ex_tokens)
            segment_ids += [sequence_b_segment_id] * self.args.bg_seq_len

        if ex_con is not None:
            # ex_con = list(set.intersection(set(ex_con), ))
            if len(ex_con) > self.args.bg_con_num:
                ex_con = ex_con[:self.args.bg_con_num]
            ex_con_len = len(ex_con)
            ex_con_pad_len = self.args.bg_con_num - ex_con_len
            ex_con += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token]*ex_con_pad_len)
            segment_ids += [sequence_b_segment_id] * self.args.bg_con_num
            input_ids += ex_con
            # find common concepts

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * concept_len + [0] * con_pad_len + \
                             [1] * ex_text_len + [0] * ex_text_pad_len + \
                             [1] * ex_con_len + [0] * ex_con_pad_len + \
                             [1] * img_len + [0] * img_padding_len
        elif att_mask_type == 'SG':
            # SG is short for Sub-Graph
            attention_mask = []
            max_len = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                      + self.args.bg_con_num + self.max_img_seq_len
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            # cc = caption concept
            cc_start, cc_end = self.max_seq_len, self.max_seq_len + concept_len
            # bc = background captions
            bc_start = self.max_seq_len + self.args.num_extra_concept
            bc_end = self.max_seq_len + self.args.num_extra_concept + ex_text_len
            # be = background extra concepts
            be_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len
            be_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + ex_con_len
            r_start = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                       + self.args.bg_con_num
            r_end = self.max_seq_len + self.args.num_extra_concept + self.args.bg_seq_len \
                     + self.args.bg_con_num + img_len
            atm = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention inside a block
            atm[c_start : c_end, c_start : c_end] = 1
            atm[l_start : l_end, l_start : l_end] = 1
            atm[cc_start : cc_end, cc_start : cc_end] = 1
            atm[bc_start : bc_end, bc_start : bc_end] = 1
            atm[be_start : be_end, be_start : be_end] = 1
            atm[r_start : r_end, r_start : r_end] = 1
            # full attention between captions and concepts
            atm[c_start : c_end, cc_start : cc_end] = 1
            atm[cc_start : cc_end, c_start : c_end] = 1
            atm[bc_start : bc_end, be_start : be_end] = 1
            atm[be_start : be_end, bc_start : bc_end] = 1
            # full attention between background concepts/captions and images
            atm[bc_start:bc_end, r_start:r_end] = 1
            atm[r_start:r_end, bc_start:bc_end] = 1
            atm[be_start:be_end, r_start:r_end] = 1
            atm[r_start:r_end, be_start:be_end] = 1
            # full attention between labels and concept/images
            atm[l_start:l_end, r_start:r_end] = 1
            atm[r_start:r_end, l_start:l_end] = 1
            atm[l_start:l_end, be_start:be_end] = 1
            atm[be_start:be_end, l_start:l_end] = 1
            atm[l_start:l_end, bc_start:bc_end] = 1
            atm[bc_start:bc_end, l_start:l_end] = 1
            atm2 = atm.clone()
            atm2[l_start:l_end, cc_start:cc_end] = 1
            atm2[cc_start:cc_end, l_start:l_end] = 1
            # caption 2 images
            atm2[c_start:c_end, r_start:r_end] = 1
            atm2[r_start:r_end, c_start:c_end] = 1
            # between concepts
            atm2[cc_start:cc_end, be_start:be_end] = 1
            atm2[be_start:be_end, cc_start:cc_end] = 1
            # add attention between images and captions/cc
            atm2[c_start:c_end, l_start:l_end] = 1
            atm2[l_start:l_end, c_start:c_end] = 1
            # add attention between captions and background concepts
            atm2[(c_start+1):c_end, be_start:be_end] = 1
            atm2[be_start:be_end, (c_start+1):c_end] = 1
            # attention_mask.append(atm)
            attention_mask = (atm, atm2)
            
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
        
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        except TypeError:
            print(input_ids)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def get_neg_txt(self, img_idx):
        if self.args.try_reranking:
            img_key = self.img_keys[img_idx]
            neg_cand = self.neg_scores[img_key]
            if isinstance(neg_cand[0][0], int):
                if self.args.less_cap_train:
                    neg_cand = [n_c for n_c in neg_cand if n_c[0]!=img_key and n_c[1] in self.sub_cap_index[n_c[0]][:self.num_captions_per_img]]
                else:
                    neg_cand = [n_c for n_c in neg_cand if n_c[0]!=img_key]
            else:
                if self.args.less_cap_train:
                    neg_cand = [n_c[0] for n_c in neg_cand if n_c[0][0]!=img_key and n_c[0][1] in self.sub_cap_index[n_c[0][0]][:self.num_captions_per_img]]
                else:
                    neg_cand = [n_c[0] for n_c in neg_cand if n_c[0][0]!=img_key]
            img_idx_neg, cap_idx_neg = random.sample(neg_cand, 1)[0]
            caption_neg = self.captions[img_idx_neg][cap_idx_neg]
            neg_extra_concept = self.get_extra_concepts(img_idx_neg, cap_idx_neg)
        else:
            img_scores = self.neg_scores['img2htxt_logit'][img_idx, :]
            sample_idx = weighted_sample(img_scores)
            neg_txt = self.neg_scores['img2htxt_index'][img_idx, sample_idx]
            # img_idx_neg = neg_txt // self.num_captions_per_img
            # cap_idx_neg = neg_txt % self.num_captions_per_img
            img_idx_neg = neg_txt // 5
            cap_idx_neg = neg_txt % 5
            caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
            neg_extra_concept = self.get_extra_concepts(self.img_keys[img_idx_neg], cap_idx_neg)
        return caption_neg, neg_extra_concept


    def get_neg_img(self, img_idx, cap_idx):
        if self.args.try_reranking:
            img_key = self.img_keys[img_idx]
            neg_cand = self.neg_scores[(img_key, cap_idx)]
            if isinstance(neg_cand[0], int):
                neg_cand = [n_c for n_c in neg_cand if n_c!=img_key]
            else:
                neg_cand = [n_c[0] for n_c in neg_cand if n_c[0]!=img_key]
            neg_img = random.sample(neg_cand, 1)[0]
            feature_neg = self.get_image(neg_img)
            od_labels_neg = self.get_od_labels(neg_img)
        else:
            cap_scores = self.neg_scores['txt2himg_logit'][img_idx*5+cap_idx, :]
            sample_idx = weighted_sample(cap_scores)
            neg_img = self.neg_scores['txt2himg_index'][img_idx*5+cap_idx, sample_idx]
            feature_neg = self.get_image(self.img_keys[neg_img])
            od_labels_neg = self.get_od_labels(self.img_keys[neg_img])
        return feature_neg, od_labels_neg, neg_img


    def __getitem__(self, index):
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            img_all_cap, img_all_con, img_con_pos, img_con_lab = self.get_image_related(img_key, cap_id=cap_idxs[1])
            if self.args.extra_concept:
                extra_concept_pos = self.get_extra_concepts(img_key, cap_idxs[1])
            else:
                extra_concept_pos = None
            example = self.tensorize_example(caption, feature, text_b=od_labels, text_c=extra_concept_pos, ex_text=img_all_cap, ex_con=img_all_con)

            # select a negative pair

            if self.args.clip_neg_sampling and random.random() <= self.args.clip_neg_prob:
                if random.random() <= 0.5:
                    caption_neg, extra_concept_neg = self.get_neg_txt(img_idx)
                    if not self.args.extra_concept:
                        extra_concept_neg = None
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, text_c=extra_concept_neg, ex_text=img_all_cap, ex_con=img_all_con)
                    neg_con_pos = [icp for icp in img_con_pos]
                    neg_con_lab = [icl for icl in img_con_lab]
                else:
                    feature_neg, od_labels_neg, neg_img_key = self.get_neg_img(img_idx, cap_idxs[1])
                    neg_all_cap, neg_all_con, neg_con_pos, neg_con_lab = self.get_image_related(neg_img_key, cap_id=-1)
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, text_c=extra_concept_pos, ex_text=neg_all_cap, ex_con=neg_all_con)
            else:
                neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random.random() <= 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    if self.args.less_cap_train:
                        cap_idx_neg = self.sub_cap_index[self.img_keys[img_idx_neg]][cap_idx_neg]
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    if self.args.extra_concept:
                        extra_concept_neg = self.get_extra_concepts(self.img_keys[img_idx_neg], cap_idx_neg)
                    else:
                        extra_concept_neg = None
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, text_c=extra_concept_neg, ex_text=img_all_cap, ex_con=img_all_con)
                    neg_con_pos = [icp for icp in img_con_pos]
                    neg_con_lab = [icl for icl in img_con_lab]
                else:
                    # randomly select a negative image 
                    feature_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    neg_all_cap, neg_all_con, neg_con_pos, neg_con_lab = self.get_image_related(self.img_keys[img_idx_neg], cap_id=-1)
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, text_c=extra_concept_pos, ex_text=neg_all_cap, ex_con=neg_all_con)

            if img_con_pos is not None:
                img_con_pos = torch.tensor(img_con_pos, dtype=torch.long)
                img_con_lab = torch.tensor(img_con_lab, dtype=torch.float)
                neg_con_pos = torch.tensor(neg_con_pos, dtype=torch.long)
                neg_con_lab = torch.tensor(neg_con_lab, dtype=torch.float)
                example_pair = tuple(list(example)+ [img_con_pos, img_con_lab] + [1] + list(example_neg) + [img_con_pos, img_con_lab] + [0])
            else:
                example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            if self.args.try_reranking and self.re_mode != 'left':
                img_idx, cap_idxs, pre_score = self.get_image_caption_index(index)
            else:
                img_idx, cap_idxs = self.get_image_caption_index(index)
            if img_idx in self.img_keys:
                img_key = img_idx
            else:
                img_key = self.img_keys[img_idx]
            # img_key = self.img_keys[img_idx]
            feature = self.get_image(img_key)
            img_all_cap, img_all_con, _, _ = self.get_image_related(img_key, cap_id=None)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            if self.args.extra_concept:
                extra_concept = self.get_extra_concepts(cap_idxs[0], cap_idxs[1])
            else:
                extra_concept = None
            example = self.tensorize_example(caption, feature, text_b=od_labels, text_c=extra_concept, ex_text=img_all_cap, ex_con=img_all_con)
            label = 1 if img_key == cap_idxs[0] else 0
            if self.args.try_reranking and self.re_mode != 'left':
                return index, tuple(list(example) + [label, pre_score])
            return index, tuple(list(example) + [label])

    def get_image(self, image_id):
        t_features = self.img_feats[image_id]
        return t_features

    def __len__(self):
        if self.is_train:
            return len(self.img_keys) * self.num_captions_per_img
        elif not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        elif self.re_mode == 'i2t': # re-ranking for images 2 text
            return len(self.img_keys) * self.args.num_captions_per_img_val
        elif self.re_mode == 't2i':
            return self.num_of_total_captions * self.args.num_images_per_cap_val
        if not self.is_train and self.args.less_cap_train and self.re_mode=='left':
            return len(self.img_keys) * self.num_captions_per_img
        else: # re-ranking for text 2 images
            return self.num_of_total_captions * self.num_images_per_cap


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_ranks(dataset, results, pr_factor=0):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    # print(np.mean(labels))
    if isinstance(results[0], tuple):
        similarities = np.array([(1-pr_factor)*results[i][0]+pr_factor*results[i][1] for i in range(len(dataset))])
    else:
        similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_image_indexs:
        num_images_per_cap = dataset.args.num_images_per_cap_val
        labels = np.reshape(labels, [-1, num_images_per_cap])
        similarities = np.reshape(similarities, [-1, num_images_per_cap])
        # print(num_images_per_cap)
        # print(similarities.shape)
        i2t_ranks, t2i_ranks = [], []
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_images_per_cap
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
        return i2t_ranks, t2i_ranks
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.args.num_captions_per_img_val
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return

def save_checkpoint_ds(model_engine, tokenizer, args, epoch, global_step):
    # a deepspeed version checkpoint saving
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    # model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            # model_to_save.save_pretrained(checkpoint_dir)
            model_engine.save_checkpoint(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            model_engine.save_fp16_model(checkpoint_dir)
            model_engine.module.config.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, model, tokenizer, val_dataset2, sub_dataset=None):
    args.train_batch_size = args.per_gpu_train_batch_size*args.n_gpu # since deepspeed is distributed
    # train_sampler = RandomSampler(train_dataset)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.n_gpu, rank=args.local_rank)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.per_gpu_train_batch_size, num_workers=args.num_workers)
    print('length of dataloader', len(train_dataloader))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(args=args, model=model, optimizer=optimizer, lr_scheduler=scheduler, training_data=train_dataset)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(args=args, model=model, optimizer=optimizer, lr_scheduler=scheduler)
    model=model_engine.module

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    sub_acc = 0.0
    # model.zero_grad()
    log_json = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        model_engine.zero_grad()
        for step, (in_index, batch) in enumerate(train_dataloader):
            # print(len(batch))
            # print(in_index)
            model_engine.train()
            # batch = tuple(t.to(model_engine.device) for t in batch)
            feat_num = len(batch)//2
            if isinstance(batch[1], tuple) or isinstance(batch[1], list):
                new_batch = []
                i = 0
                for t in batch:
                    if i == 1 or i == 1+feat_num:
                        new_batch.append([s.to(model_engine.device) for s in t])
                    else:
                        new_batch.append(t.to(model_engine.device))
                    i += 1
                batch = new_batch
                inputs = {
                    'input_ids':      torch.cat((batch[0], batch[feat_num]), dim=0),
                    'token_type_ids': torch.cat((batch[2], batch[feat_num+2]), dim=0),
                    'img_feats':      torch.cat((batch[3], batch[feat_num+3]), dim=0),
                    'labels':         torch.cat((batch[6], batch[feat_num+6]), dim=0),
                    'con_pos':        torch.cat((batch[4], batch[feat_num+4]), dim=0),
                    'con_lab':        torch.cat((batch[5], batch[feat_num+5]), dim=0)
                }
                inputs['attention_mask']=[]
                for i in range(len(batch[1])):
                    inputs['attention_mask'].append(torch.cat((batch[1][i], batch[feat_num+1][i]), dim=0))
            else:
                batch = tuple(t.to(model_engine.device) for t in batch)
                inputs = {
                    'input_ids':      torch.cat((batch[0], batch[5]), dim=0),
                    'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
                    'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),
                    'img_feats':      torch.cat((batch[3], batch[8]), dim=0),
                    'labels':         torch.cat((batch[4], batch[9]), dim=0)
                }
            inputs = prepare_inputs(inputs, args)
            inputs['con_lambda'] = args.con_lambda
            outputs = model(**inputs)
            loss, logits, con_logits = outputs[:3]
            # print('out model batch size', logits.shape)
            """ # no need for deepspeed here!
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            """
            model_engine.backward(loss)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_score = compute_score_with_logits(logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.per_gpu_train_batch_size * 2)
            batch_sub_score = torch.sum((con_logits >= 0.5)==inputs['con_lab'].reshape(-1))
            batch_sub_acc = batch_sub_score.item() / (args.per_gpu_train_batch_size*args.neg_con_detect*2)
            sub_acc += batch_sub_acc
            assert(args.neg_con_detect==inputs['con_lab'].shape[1])
            global_loss += loss.item()
            global_acc += batch_acc
            model_engine.step()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # scheduler.step()
                # optimizer.step()
                model_engine.zero_grad()
                # model_engine.step()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f}), sub-score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        batch_acc, global_acc / global_step, batch_sub_acc, sub_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint_ds(model_engine, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training:
                        if args.less_cap_train:
                            logger.info('Perform left cpation evaluation')
                            acc = acc_test(args, model_engine, sub_dataset)
                            logger.info('accuracy on the left captions %s' % (acc))
                        if args.try_reranking:
                            logger.info("Perform evaluation (with pre_factor iteration) at step: %d" % (global_step))
                            best_pr = -1
                            best_recall = None
                            best_rsum = -1
                            logit_test_result_i2t = test(args, model_engine, val_dataset)
                            if val_dataset2 is not None:
                                logit_t2i_result = test(args, model_engine, val_dataset2)
                                # print(t2i_result)
                            for pr_f in range(10):
                                if pr_f != 0:
                                    continue
                                logger.info('performing evaluation with pre-factor %s' % (pr_f*0.1))
                                eval_result = evaluate(val_dataset, logit_test_result_i2t, pr_factor=pr_f*0.1)
                                rank_accs = eval_result['i2t_retrieval']
                                if rank_accs['R@1'] > best_score:
                                    best_score = rank_accs['R@1']
                                epoch_log = {'pr_f': pr_f*0.1, 'epoch': epoch, 'global_step': global_step, 
                                            'i2t_R1': rank_accs['R@1'], 'i2t_R5': rank_accs['R@5'], 
                                            'i2t_R10': rank_accs['R@10'], 'best_R1':best_score}
                                rsum = rank_accs['R@1']+rank_accs['R@5']+rank_accs['R@10']
                                if val_dataset2 is not None:
                                    t2i_result = evaluate(val_dataset2, logit_t2i_result, pr_factor=pr_f*0.1)
                                    rank_accs = t2i_result['t2i_retrieval']
                                    t2i_log = {'t2i_R1': rank_accs['R@1'], 't2i_R5': rank_accs['R@5'], 
                                            't2i_R10': rank_accs['R@10']}
                                    rsum += rank_accs['R@1'] + rank_accs['R@5'] + rank_accs['R@10']
                                    epoch_log.update(t2i_log)
                                if rsum > best_rsum:
                                    best_rsum = rsum
                                    best_pr = 0.1*pr_f
                                    best_recall = epoch_log
                            logger.info("best rsum got %s with pre-factor %s" % (best_rsum*100, best_pr))
                            best_recall['rsum'] = best_rsum
                            if args.less_cap_train:
                                best_recall['left_cap_acc'] = acc
                            log_json.append(best_recall)
                            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                                json.dump(log_json, fp) 
                        else:
                            logger.info("Perform evaluation at step: %d" % (global_step))
                            test_result = test(args, model_engine, val_dataset)
                            eval_result = evaluate(val_dataset, test_result)
                            rank_accs = eval_result['i2t_retrieval']
                            if rank_accs['R@1'] > best_score:
                                best_score = rank_accs['R@1']
                            epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                        'i2t_R1': rank_accs['R@1'], 'i2t_R5': rank_accs['R@5'], 
                                        'i2t_R10': rank_accs['R@10'], 'best_R1':best_score}
                            rsum = rank_accs['R@1']+rank_accs['R@5']+rank_accs['R@10']
                            if val_dataset2 is not None:
                                t2i_result = test(args, model_engine, val_dataset2)
                                t2i_result = evaluate(val_dataset2, t2i_result)
                                rank_accs = t2i_result['t2i_retrieval']
                                t2i_log = {'t2i_R1': rank_accs['R@1'], 't2i_R5': rank_accs['R@5'], 
                                        't2i_R10': rank_accs['R@10']}
                                epoch_log.update(t2i_log)
                                rsum += rank_accs['R@1'] + rank_accs['R@5'] + rank_accs['R@10']
                            epoch_log['rsum'] = rsum
                            if args.less_cap_train:
                                epoch_log['left_cap_acc'] = acc
                            log_json.append(epoch_log)
                            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                                json.dump(log_json, fp) 
    return global_step, global_loss / global_step


def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(eval_dataset)
    eval_sampler = SequentialDistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    # for indexs, batch in tqdm(eval_dataloader):
    for indexs, batch in tqdm(eval_dataloader):
        g_indexs = indexs.to(model.device)
        if isinstance(batch[1], tuple) or isinstance(batch[1], list):
            new_batch = []
            i = 0
            for t in batch:
                if i == 1:
                    new_batch.append([s.to(model.device) for s in t])
                else:
                    new_batch.append(t.to(model.device))
                i += 1
            batch = new_batch
        else:
            batch = [t.to(model.device) for t in batch]
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4]
            }
            inputs = prepare_inputs(inputs, args)
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            # print(result.shape)
            result = distributed_concat(result.contiguous())
            # print(result.shape)
            g_indexs = distributed_concat(g_indexs)
            g_indexs = [_.to(torch.device("cpu")) for _ in g_indexs]
            result = [_.to(torch.device("cpu")) for _ in result]
            if args.try_reranking:
                g_pre_scores = distributed_concat(batch[5])
                results.update({idx.item(): (res.item(), ps.item()) for idx, res, ps in zip(g_indexs, result, g_pre_scores)})
            else:
                results.update({idx.item(): res.item() for idx, res in zip(g_indexs, result)})
    return results

def acc_test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # eval_sampler = SequentialSampler(eval_dataset)
    eval_sampler = SequentialDistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.per_gpu_eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    # for indexs, batch in tqdm(eval_dataloader):
    for indexs, batch in tqdm(eval_dataloader):
        g_indexs = indexs.to(model.device)
        if isinstance(batch[1], list) or isinstance(batch[1], tuple):
            new_batch = []
            i = 0
            for t in batch:
                if i == 1:
                    new_batch.append([s.to(model.device) for s in t])
                else:
                    new_batch.append(t.to(model.device))
                i += 1
            batch = new_batch
        else:
            batch = [t.to(model.device) for t in batch]
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4]
            }
            inputs = prepare_inputs(inputs, args)
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            # print(result.shape)
            result = compute_score_with_logits(logits, inputs['labels'])
            result = distributed_concat(result.contiguous())
            # print(result.shape)
            g_indexs = distributed_concat(g_indexs)
            g_indexs = [_.to(torch.device("cpu")) for _ in g_indexs]
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(g_indexs, result)})
    return sum([v for k,v in results.items()])/len(results)

def evaluate(eval_dataset, test_results, pr_factor=0):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results, pr_factor)
    rank = [1, 5, 10]
    eval_result = {}
    if i2t_ranks:
        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
        logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    i2t_accs[0], i2t_accs[1], i2t_accs[2]))
        eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc))) 


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps', 'extra_concept', 'num_extra_concept']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def prepare_inputs(inputs, args):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if inputs[k].dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                inputs[k]=v.to(dtype=args.dtype)
        if isinstance(v, list) or isinstance(v, tuple):
            new_v = []
            for vv in v:
                if vv.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                    new_v.append(vv.to(dtype=args.dtype))
                else:
                    new_v.append(vv)
            inputs[k] = new_v
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset" 
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, 
                        help="image key tsv to select a subset of images for evaluation. "
                        "This is useful in 5-folds evaluation. The topn index file is not " 
                        "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str, 
                        help="index of a list of (img_key, cap_idx) for each image."
                        "this is used to perform re-rank using hard negative samples."
                        "useful for validation set to monitor the performance during training.")
    parser.add_argument("--eval_image_index_file", default='', type=str, 
                        help="index of a list of (img_key, cap_idx) for each image."
                        "this is used to perform re-rank using hard negative samples."
                        "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true', 
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, 
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str, 
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                        "C: caption, L: labels, R: image regions; CLR is full attention by default."
                        "CL means attention between caption and labels."
                        "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument("--num_images_per_cap_val", default=5, type=int,
                        help="number of images for each testing caption.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument('--extra_concept', action='store_true', help="Whether to add more related concepts from the concept graph.")
    parser.add_argument('--num_extra_concept', type=int, default=-1, help="Number of extra concapts added")
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help="Which GPUs to use")
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument('--clip_neg_sampling', action='store_true', help="Whether to use clip-based scores for negative sampling")
    parser.add_argument('--clip_neg_prob', type=float, default=0.5, help='probability of performing clip neg sampling')
    parser.add_argument('--do_cp2model', action='store_true', help="Whether to transform zero checkpoint to model")
    parser.add_argument('--print_zeroshot', action='store_true', help="Whether to print the zero-shot results")
    parser.add_argument('--concept2id_file', type=str, default='', help="the file name storing the mapping"
                        "between extra concept and their token ids")
    parser.add_argument('--try_reranking', action='store_true', help='whether to try pr factor')
    parser.add_argument('--test_prf', type=float, default=0, help='pr factor used for testing')
    parser.add_argument('--less_cap_train', action='store_true', help='if use less captions to train and left caption to eval')
    parser.add_argument('--train_sub_index', type=str, default='', help='pre-ranking index file for sub caption retrieval')
    parser.add_argument('--bg_seq_len', type=int, default=80, help='maximal background caption length')
    parser.add_argument('--bg_con_num', type=int, default=50, help='maximal number of background concepts')
    parser.add_argument('--include_noise', action='store_true', help='whether to add noise in background')
    parser.add_argument('--neg_con_detect', type=int, default=0, help='whether to add a token level concept detection task')
    parser.add_argument('--con_lambda', type=float, default=0, help='lambda for concept loss')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    # set the args.dtype for fp16
    if args.deepspeed_config is not None:
        with open(args.deepspeed_config, 'r') as of:
            ds_config = json.load(of)
        if 'fp16' in ds_config:
            if ds_config['fp16']['enabled']:
                args.dtype = torch.float16
            else:
                args.dtype = torch.float32

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification2
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        if args.extra_concept and args.concept2id_file:
            concept2id = torch.load(op.join(args.data_dir, args.concept2id_file))
            num_extra_concept = len(concept2id)
            if '[UNK]' in concept2id:
                num_extra_concept -= 1
            args.vocab_size = config.vocab_size
            args.concept_size = num_extra_concept
            config.vocab_size += num_extra_concept
            del(concept2id)
        model = model_class.from_pretrained(args.model_name_or_path, 
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        config.save_pretrained(args.output_dir)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        # config = config_class.from_pretrained(args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        config = config_class.from_pretrained(checkpoint)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    # write num_layers into args
    args.num_layers = config.num_hidden_layers

    # model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_img_feat = torch.load(op.join(args.data_dir, 'train_img_{}_feats.pt'.format(args.img_feature_type)))
        logger.info('Creating training set')
        train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True, img_feat=train_img_feat)
        if args.evaluate_during_training:
            logger.info('Creating evaluation set')
            if 'coco_ir' not in args.data_dir:
                val_split = 'val'
            else:
                val_split = 'minival'
            if args.less_cap_train:
                # train_img_feat = None
                val_split = 'train'
                # train_img_feat = torch.load(op.join(args.data_dir, 'train_img_{}_feats.pt'.format(args.img_feature_type)))
            else:
                train_img_feat = None
            val_dataset = RetrievalDataset(tokenizer, args, val_split, is_train=False, reranking_mode='i2t', img_feat=train_img_feat)
            if args.eval_image_index_file:
                val_dataset2 = RetrievalDataset(tokenizer, args, val_split, is_train=False, reranking_mode='t2i', img_feat=train_img_feat)
            else:
                val_dataset2 = None
            if args.less_cap_train:
                sub_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=False, reranking_mode='left', img_feat=train_img_feat)
            else:
                sub_dataset = None
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer, val_dataset2, sub_dataset)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
        if args.eval_image_index_file:
            test_dataset2 = RetrievalDataset(tokenizer, args, args.test_split, is_train=False, reranking_mode='t2i')
        else:
            test_dataset2 = None
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        if args.do_cp2model:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=checkpoint)
            model.load_state_dict(state_dict)
            model.save_pretrained(checkpoint)
            return 1
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not \
                any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if \
                any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.scheduler == "constant":
            scheduler = WarmupConstantSchedule(
                    optimizer, warmup_steps=args.warmup_steps)
        elif args.scheduler == "linear":
            scheduler = WarmupLinearSchedule(
                    optimizer, warmup_steps=args.warmup_steps, t_total=100)
        model_engine, _, _, _= deepspeed.initialize(args, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        if not args.print_zeroshot:
            model_engine.load_checkpoint(checkpoint)
        # model.to(args.device)
        # if args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        pred_file = get_predict_file(args)
        if op.isfile(pred_file):
            logger.info("Prediction file exist, skip inference.")
            if args.do_eval:
                test_result = torch.load(pred_file)
        else:
            test_result = test(args, model_engine, test_dataset)
            torch.save(test_result, pred_file)
            logger.info("Prediction results saved to {}.".format(pred_file))
        if test_dataset2:
            t2i_result = test(args, model_engine, test_dataset2)

        if args.do_eval:
            eval_result = evaluate(test_dataset, test_result, pr_factor=args.test_prf)
            if test_dataset2:
                t2i_result = evaluate(test_dataset2, t2i_result, pr_factor=args.test_prf)
                eval_result.update(t2i_result)
            result_file = op.splitext(pred_file)[0] + '.eval.json'
            with open(result_file, 'w') as f:
                json.dump(eval_result, f)
            logger.info("Evaluation results saved to {}.".format(result_file))

def main_clip():
    clip_path = '/remote-home/zjli/vis_text/pre_trained_ViT_32B.pth'
    test_clip(clip_path)

if __name__ == "__main__":
    # print(123)
    main()
