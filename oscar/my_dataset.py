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
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification, test_clip
from transformers.pytorch_transformers import BertTokenizer, BertConfig 
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from oscar.sequential_eval_utils import *


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True, reranking_mode='i2t'):
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
        assert reranking_mode in ['i2t', 't2i']
        self.re_mode = reranking_mode
        self.img_file = args.img_feat_file
        caption_file = op.join(args.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        self.num_of_total_captions = sum([len(self.captions[k]) for k in self.img_keys])

        # get the image image_id to index map
        # imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        # self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
        # get the image features and labels
        img_feat_file = op.join(args.data_dir, '{}_img_{}_feats.pt'.format(split, args.img_feature_type))
        self.img_feats = torch.load(img_feat_file)

        # get extra concepts
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
        
        if args.add_od_labels:
            labels_file = op.join(args.data_dir, '{}_{}_labels.pt'.format(split, args.od_label_type))
            self.labels = torch.load(labels_file)

        if args.clip_neg_sampling and is_train:
            neg_scpres_file = op.join(args.data_dir, '{}_clip_ft_scores.pt'.format(split))
            if args.try_reranking:
                neg_scpres_file = op.join(args.data_dir, '{}_pre_ranking2.pt'.format(split))
            self.neg_scores = torch.load(neg_scpres_file)

        if is_train:
            self.num_captions_per_img = args.num_captions_per_img_train
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
                image_index_file = op.join(args.data_dir, args.eval_image_index_file)
                self.image_indexs = torch.load(image_index_file)
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
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            if self.args.try_reranking:
                (img_key1, cap_idx1), pre_score = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
                return img_idx, [img_key1, cap_idx1], pre_score
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        if not self.is_train and self.has_image_indexs:
            cap_idx = index // self.num_images_per_cap
            cap_img_idx = cap_idx // 5
            cap_cap_idx = cap_idx % 5
            img_idx = index % self.num_images_per_cap
            if self.args.try_reranking:
                img_key1, pre_score = self.image_indexs[(self.img_keys[cap_img_idx],cap_cap_idx)][img_idx]
                return img_key1, [self.img_keys[cap_img_idx], cap_cap_idx], pre_score
            img_key1 = self.image_indexs[(self.img_keys[cap_img_idx],cap_cap_idx)][img_idx]
            return img_key1, [self.img_keys[cap_img_idx], cap_cap_idx]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
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

    def tensorize_example3(self, text_a, img_feat, text_b=None, text_c=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
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
                             [1] * img_len + [0] * img_padding_len
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
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def tensorize_example(self, text_a, img_feat, text_b=None, text_c=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
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
            segment_ids += additional_segment_ids
        
        if text_c is not None:
            if len(text_c) > self.max_seq_len-len(tokens):
                text_c = text_c[:(self.max_seq_len-len(tokens))]
            concept_len = len(text_c)
            segment_ids = segment_ids[:seq_a_len] + [sequence_c_segment_id]*concept_len + segment_ids[seq_a_len:]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = tokens[:seq_a_len] + text_c + tokens[seq_a_len:]

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * seq_padding_len)
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = tokens

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
                             [1] * img_len + [0] * img_padding_len
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
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def tensorize_example2(self, text_a, img_feat, text_b=None, text_c=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
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
            if text_c:
                tmp_len = len(tokens_b)
                if type(text_c) == str:
                    tokens_b += self.tokenizer.tokenize(text_c)
                    tokens_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
                else:
                    tokens_b = self.tokenizer.convert_tokens_to_ids(tokens_b) + text_c
            else:
                # print('None' if text_c is None else 'empty string')
                tokens_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                num_left_tokens = max(0, self.max_seq_len - len(tokens) - 1) # to avoid -1
                assert(num_left_tokens >= 0)
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            # tokens += tokens_b + self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
            tokens = self.tokenizer.convert_tokens_to_ids(tokens) + tokens_b \
                        + self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])
            additional_segment_ids = [sequence_b_segment_id] * (len(tokens_b) + 1)
            if text_c:
                if tmp_len <= self.max_seq_len - len(tokens) - 1:
                    for idx in range(tmp_len, len(additional_segment_ids)-1):
                        additional_segment_ids[idx] = sequence_c_segment_id
            segment_ids += additional_segment_ids

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token] * seq_padding_len)
        segment_ids += [pad_token_segment_id] * seq_padding_len
        if type(tokens[0]) == str:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = tokens

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
                             [1] * img_len + [0] * img_padding_len
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
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def get_neg_txt(self, img_idx):
        if self.args.try_reranking:
            img_key = self.img_keys[img_idx]
            neg_cand = self.neg_scores[img_key]
            if isinstance(neg_cand[0][0], int):
                neg_cand = [n_c for n_c in neg_cand if n_c[0]!=img_key]
            else:
                neg_cand = [n_c[0] for n_c in neg_cand if n_c[0][0]!=img_key]
            img_idx_neg, cap_idx_neg = random.sample(neg_cand, 1)[0]
            caption_neg = self.captions[img_idx_neg][cap_idx_neg]
            neg_extra_concept = self.get_extra_concepts(img_idx_neg, cap_idx_neg)
        else:
            img_scores = self.neg_scores['img2htxt_logit'][img_idx, :]
            sample_idx = weighted_sample(img_scores)
            neg_txt = self.neg_scores['img2htxt_index'][img_idx, sample_idx]
            img_idx_neg = neg_txt // self.num_captions_per_img
            cap_idx_neg = neg_txt % self.num_captions_per_img
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
        return feature_neg, od_labels_neg


    def __getitem__(self, index):
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.img_keys[img_idx]
            feature = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            if self.args.extra_concept:
                extra_concept_pos = self.get_extra_concepts(img_key, cap_idxs[1])
            else:
                extra_concept_pos = None
            example = self.tensorize_example(caption, feature, text_b=od_labels, text_c=extra_concept_pos)

            # select a negative pair

            if self.args.clip_neg_sampling and random.random() <= self.args.clip_neg_prob:
                if random.random() <= 0.5:
                    caption_neg, extra_concept_neg = self.get_neg_txt(img_idx)
                    if not self.args.extra_concept:
                        extra_concept_neg = None
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, text_c=extra_concept_neg)
                else:
                    feature_neg, od_labels_neg = self.get_neg_img(img_idx, cap_idxs[1])
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, text_c=extra_concept_pos)
            else:
                neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1, len(self.img_keys)))
                img_idx_neg = random.choice(neg_img_indexs)
                if random.random() <= 0.5:
                    # randomly select a negative caption from a different image.
                    cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                    caption_neg = self.captions[self.img_keys[img_idx_neg]][cap_idx_neg]
                    if self.args.extra_concept:
                        extra_concept_neg = self.get_extra_concepts(self.img_keys[img_idx_neg], cap_idx_neg)
                    else:
                        extra_concept_neg = None
                    example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels, text_c=extra_concept_neg)
                else:
                    # randomly select a negative image 
                    feature_neg = self.get_image(self.img_keys[img_idx_neg])
                    od_labels_neg = self.get_od_labels(self.img_keys[img_idx_neg])
                    example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg, text_c=extra_concept_pos)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            if self.args.try_reranking:
                img_idx, cap_idxs, pre_score = self.get_image_caption_index(index)
            else:
                img_idx, cap_idxs = self.get_image_caption_index(index)
            if img_idx in self.img_keys:
                img_key = img_idx
            else:
                img_key = self.img_keys[img_idx]
            # img_key = self.img_keys[img_idx]
            feature = self.get_image(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_labels = self.get_od_labels(img_key)
            if self.args.extra_concept:
                extra_concept = self.get_extra_concepts(cap_idxs[0], cap_idxs[1])
            else:
                extra_concept = None
            example = self.tensorize_example(caption, feature, text_b=od_labels, text_c=extra_concept)
            label = 1 if img_key == cap_idxs[0] else 0
            if self.args.try_reranking:
                return index, tuple(list(example) + [label, pre_score])
            return index, tuple(list(example) + [label])

    def get_image(self, image_id):
        t_features = self.img_feats[image_id]
        return t_features

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.img_keys) ** 2 * self.num_captions_per_img
        if self.re_mode == 'i2t': # re-ranking for images 2 text
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
        num_images_per_cap = dataset.num_images_per_cap
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
        num_captions_per_img = dataset.num_captions_per_img
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


def train(args, train_dataset, val_dataset, model, tokenizer, val_dataset2):
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
    # model.zero_grad()
    log_json = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        model_engine.zero_grad()
        for step, (in_index, batch) in enumerate(train_dataloader):
            # print(in_index)
            model_engine.train()
            batch = tuple(t.to(model_engine.device) for t in batch)
            inputs = {
                'input_ids':      torch.cat((batch[0], batch[5]), dim=0),
                'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
                'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),
                'img_feats':      torch.cat((batch[3], batch[8]), dim=0),
                'labels':         torch.cat((batch[4], batch[9]), dim=0)
            }
            inputs = prepare_inputs(inputs, args)
            outputs = model(**inputs)
            loss, logits = outputs[:2]
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
                        "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        batch_acc, global_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint_ds(model_engine, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training:
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
                            if val_dataset2 is not None:
                                t2i_result = test(args, model_engine, val_dataset2)
                                t2i_result = evaluate(val_dataset2, t2i_result)
                                rank_accs = t2i_result['t2i_retrieval']
                                t2i_log = {'t2i_R1': rank_accs['R@1'], 't2i_R5': rank_accs['R@5'], 
                                        't2i_R10': rank_accs['R@10']}
                                epoch_log.update(t2i_log)
                            log_json.append(epoch_log)
                            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                                json.dump(log_json, fp) 
    return global_step, global_loss / global_step