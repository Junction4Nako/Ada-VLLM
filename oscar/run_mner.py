# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy, time, json
import base64
import random

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import _pickle as cPickle
import os.path as op
from tqdm import tqdm
import jsonlines

from oscar.utils.tsv_file import TSVFile
from oscar.modeling.modeling_vlbert import BiImageBertForNER, BiImageBertForSequenceClassification, BiImageBertForSequenceClassificationPlus
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

from oscar.utils.misc import set_seed, mkdir
from oscar.utils.logger import setup_logger
from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)


MODEL_CLASSES = {
    'bert': (BertConfig, BiImageBertForNER, BertTokenizer),
}

label_mapping = {'entailment':2,'neutral':1,'contradiction':0}
log_json = []
debug_size = 500

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# processor = MNERProcessor()
labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
label_map = {label:i for i,label in enumerate(labels, 1)}
ans2label_map = {i:label for i,label in enumerate(labels, 1)}
ans2label_map[0] = "PAD"


def _load_dataset(args, name):
    if name in ['train', 'val', 'test']:
        if name == 'val':
            c_name = 'dev'
        else:
            c_name = name
        examples = [line for line in jsonlines.open(op.join(args.txt_data_dir, 'snli_ve_{}.jsonl'.format(c_name)))]
        sgs = json.load(open(op.join(args.txt_data_dir, '{}_sgs.json'.format(name)), 'r'))
    elif name == 'train+val':
        examples = []
        sgs = []
        for name in ['train', 'val']:
            c_name = 'dev' if name == 'val' else 'train'
            examples.extend([line for line in jsonlines.open(op.join(args.txt_data_dir, 'snli_ve_{}.jsonl'.format(c_name)))])
            sgs.extend(json.load(open(op.join(args.txt_data_dir, '{}_sgs.json'.format(name)), 'r')))
    else:
        raise NotImplementedError

    return examples, sgs

class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []

    print("The number of samples: "+ str(len(data)))
    return data


def mmreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label= []
    auxlabel = []
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1]+'.jpg'
            continue
        if line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) >0:
        data.append((sentence,label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: "+ str(len(data)))
    print("The number of images: "+ str(len(imgs)))
    return data, imgs, auxlabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def _read_mmtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return mmreadfile(input_file)


class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "valid.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
        return examples

class NERDataset(Dataset):
    """ SNLI-VE Dataset """

    def __init__(self, args, name, tokenizer):
        super(NERDataset, self).__init__()
        assert name in ['train', 'dev', 'test']

        self.args = args
        self.name = name
        processor = MNERProcessor()
        self.label_list = processor.get_labels()
        self.num_labels = len(self.label_list) + 1

        # load image features
        t_start = time.time()
        self.img_feature_file = None
        self.img_feat_offset_map = None

        self.id2phrase = json.load(open(args.bivinvl_id2phrase, 'r'))
        self.phrase2id = {tuple(v):int(k) for k,v in self.id2phrase.items()}

        if args.img_feature_type == 'faster_r-cnn':
            if args.img_feat_format == 'pt':
                self.img_feat = torch.load(op.join(args.data_dir, 'img_frcnn_obj_feats.pt'))
                if args.add_od_labels:
                    self.od_labels = torch.load(op.join(args.data_dir, 'img_od_tags.pt'))
            elif args.img_feat_format == 'tsv':
                # self.load_img_tsv_features()
                self.img_file = op.join(args.img_feat_dir, 'features.tsv')
                self.img_tsv = TSVFile(self.img_file)
                imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
                self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
                self.image_id2idx = {str(int(k.split('_')[-1])):v for k,v in self.image_id2idx.items()}
            
                if args.add_od_labels:
                    label_data_dir = op.dirname(self.img_file)
                    label_file = os.path.join(label_data_dir, "predictions.tsv")
                    self.label_tsv = TSVFile(label_file)
                    self.od_labels = {}
                    for line_no in tqdm(range(self.label_tsv.num_rows())):
                        row = self.label_tsv.seek(line_no)
                        image_id = row[0]
                        if '_' in image_id:
                            image_id = image_id.split('_')[-1]
                        results = json.loads(row[1])
                        objects = results['objects'] if type(
                            results) == dict else results
                        self.od_labels[int(image_id)] = {
                            "image_h": results["image_h"] if type(
                                results) == dict else 600,
                            "image_w": results["image_w"] if type(
                                results) == dict else 800,
                            "class": [cur_d['class'] for cur_d in objects],
                            "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                            dtype=np.float32)
                        }
                    self.label_tsv._fp.close()
                    self.label_tsv._fp = None
        else:
            raise NotImplementedError

        t_end = time.time()
        self.c_ids = 0
        logger.info('Info: loading {0} features using {1:.2f} secs'.format(name, (t_end-t_start)))

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer

        if name == 'train':
            self.examples = processor.get_train_examples(args.data_dir)
        elif name == 'dev':
            self.examples = processor.get_dev_examples(args.data_dir)
        elif name == 'test':
            self.examples = processor.get_test_examples(args.data_dir)

        self.label_map = label_mapping

        # loading the phrases concepts
        # self.phrases = json.load(open(os.path.join(args.data_dir, '{}_sg_tuples.json'.format(name)), 'r'))
        # assert len(self.phrases)==len(self.examples), 'phrases length {} but example length {}'.format(len(self.phrases), len(self.examples))

        logger.info('%s Data Examples: %d' % (name, len(self.examples)))

    def tensorize_example(self, example, phrases, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        text_a = example.text_a
        labellist = example.label
        text_list = text_a.split(' ')
        tokens_a = []
        labels = []
        for i, word in enumerate(text_list):
            token = self.tokenizer.tokenize(word)
            tokens_a.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        img_key = example.img_id.split('.')[0]
        num_extra_tokens = 2
        num_phrases = self.args.max_phrases
        if len(tokens_a) > self.args.max_seq_length - num_extra_tokens:
            tokens_a = tokens_a[:(self.args.max_seq_length-num_extra_tokens)]
            labels = labels[:(self.args.max_seq_length-num_extra_tokens)]

        # phrase_nodes, phrase_info = phrases
        phrase_nodes = []
        phrase_nodes = [self.phrase2id[tuple(p)] for p in phrase_nodes if tuple(p) in self.phrase2id]
        # assert 'vqa' in phrase_info and int(phrase_info.split('_')[1
        if len(phrase_nodes) > num_phrases + self.args.max_seq_length - 2 - len(tokens_a):
            phrase_nodes = phrase_nodes[:(num_phrases+self.args.max_seq_length-2-len(tokens_a))]

        seq_tokens_a = [self.tokenizer.cls_token,] + tokens_a
        label_ids = [label_map["[CLS]"]]
        input_ids_a = self.tokenizer.convert_tokens_to_ids(seq_tokens_a) + phrase_nodes + [self.tokenizer.vocab[self.tokenizer.sep_token]]
        label_ids = label_ids + [label_map[t] for t in labels] + [0]*len(phrase_nodes) + [label_map['[SEP]']]
        segment_ids_a = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + len(phrase_nodes) + 1)
        seq_a_len = len(input_ids_a)
        input_mask_a = [1] * len(input_ids_a)
        
        tokens_b = self.get_od_labels(img_key)
        tokens_b = self.tokenizer.tokenize(tokens_b)

        if self.args.use_b_text:
            # add object tags in sequence a to add more information!
            if len(tokens_b) > num_phrases + self.args.max_seq_length - len(input_ids_a):
                add_tokensb_b = tokens_b[:(self.args.max_seq_length + self.args.max_phrases - len(input_ids_a))]
            else:
                add_tokensb_b = tokens_b
            input_ids_a += self.tokenizer.convert_tokens_to_ids(add_tokensb_b)
            input_mask_a += [1] * len(add_tokensb_b)
            segment_ids_a += [sequence_b_segment_id] * len(add_tokensb_b)
            seq_a_len += len(add_tokensb_b)

        if len(tokens_b) > self.args.max_tag_length - 2:
            tokens_b = tokens_b[: (self.args.max_tag_length-2)]
        seq_tokens_b = [self.tokenizer.cls_token,] + tokens_b + [self.tokenizer.sep_token]
        input_ids_b = self.tokenizer.convert_tokens_to_ids(seq_tokens_b)
        segment_ids_b = [sequence_b_segment_id] * len(seq_tokens_b)
        input_mask_b = [1] * len(input_ids_b)
        seq_b_len = len(input_ids_b)

        # padding sequences
        seq_len_a = len(input_ids_a)
        tmp_max_seq_len = self.args.max_seq_length + self.args.max_phrases
        seq_padding_len_a = tmp_max_seq_len - seq_len_a
        input_ids_a += seq_padding_len_a * [0,]
        input_mask_a += seq_padding_len_a * [0,]
        segment_ids_a += seq_padding_len_a * [pad_token_segment_id,]
        label_ids += [0] * seq_padding_len_a

        seq_padding_len_b = self.args.max_tag_length - seq_b_len
        input_ids_b += seq_padding_len_b * [0, ]
        input_mask_b += seq_padding_len_b * [0, ]
        segment_ids_b += seq_padding_len_b * [pad_token_segment_id, ]

        assert len(input_ids_a) == self.args.max_seq_length + self.args.max_phrases
        assert len(input_mask_a) == self.args.max_seq_length + self.args.max_phrases
        assert len(segment_ids_a) == self.args.max_seq_length + self.args.max_phrases
        assert len(label_ids) == self.args.max_seq_length + self.args.max_phrases
        assert len(input_ids_b) == self.args.max_tag_length
        assert len(input_mask_b) == self.args.max_tag_length
        assert len(segment_ids_b) == self.args.max_tag_length


        # image features
        if self.args.img_feat_format == 'pt':
            img_feat = self.img_feat[img_key] #[:, 0:self.args.img_feature_dim]  # torch
        elif self.args.img_feat_format == 'tsv':
            img_feat = self.get_img_feature(str(img_key))
            # img_feat = torch.from_numpy(img_features)
        else:
            raise NotImplementedError

        if img_feat.shape[0] > self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            input_mask_b = input_mask_b + [1] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            input_mask_b = input_mask_b + [1] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            input_mask_b = input_mask_b + ([0] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]


        if self.c_ids < 1 and 'train' in self.name:
            self.c_ids += 1
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_a: %s" % " ".join([str(x) for x in seq_tokens_a]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
            logger.info("input_mask_a: %s" % " ".join([str(x) for x in input_mask_a]))
            logger.info("segment_ids_a: %s" % " ".join([str(x) for x in segment_ids_a]))
            logger.info("tokens_b: %s" % " ".join([str(x) for x in seq_tokens_b]))
            logger.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
            logger.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
            logger.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))


        return (torch.tensor(input_ids_a, dtype=torch.long),
                torch.tensor(input_mask_a, dtype=torch.long),
                torch.tensor(segment_ids_a, dtype=torch.long),
                torch.tensor(input_ids_b, dtype=torch.long),
                torch.tensor(input_mask_b, dtype=torch.long),
                torch.tensor(segment_ids_b, dtype=torch.long),
                torch.tensor(label_ids, dtype=torch.long),
                img_feat)

    def __getitem__(self, index):
        if self.args.load_fast:
            raise NotImplementedError
        else:
            entry = self.examples[index]
            example = self.tensorize_example(entry, None,
                cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return index, example

    def get_label(self, index):
        return label_mapping[self.examples[index]['gold_label']]

    def get_pairid(self, index):
        return self.examples[index]['pairID']

    def __len__(self):
        return len(self.examples)

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def get_od_labels(self, img_key):
        if self.args.img_feat_format == 'tsv':
            if type(self.od_labels[img_key]) == str:
                od_labels = self.od_labels[img_key]
            else:
                od_labels = ' '.join(list(set(self.od_labels[img_key]['class'])))
        else:
            od_labels = ' '.join(list(set(self.od_labels[str(img_key)])))
        return od_labels

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            img_feature_path = os.path.join(self.args.img_feat_dir, '{}_img_frcnn_feats.tsv'.format(self.name))
            t_s = time.time()
            self.img_feature_file = open(img_feature_path, 'r')
            t_e = time.time()
            logger.info("Open {} image time: {}".format(self.name, (t_e - t_s)))

    def check_img_feature_offset_map(self):
        """ load the image feature offset map """
        if self.img_feat_offset_map is None:
            img_feature_path = os.path.join(self.args.img_feat_dir, '{}_img_frcnn_feats_offset_map.json'.format(self.name))
            t_s = time.time()
            self.img_feat_offset_map = json.load(open(img_feature_path))
            t_e = time.time()
            logger.info("Load {} images: {}, time: {}".format(self.name, len(self.img_feat_offset_map), (t_e - t_s)))

    def get_img_feature(self, image_id):
        if self.args.img_feat_format == 'tsv':
            image_idx = self.image_id2idx[str(image_id)]
            row = self.img_tsv.seek(image_idx)
            num_boxes = int(row[1])
            features = np.frombuffer(base64.b64decode(row[-1]),
                                    dtype=np.float32).reshape((num_boxes, -1))
            if not features.flags['WRITEABLE']:
                features = np.copy(features)
            t_features = torch.from_numpy(features)
        elif self.args.img_feat_format == 'pt':
            t_features = self.img_feat[str(image_id)]
        return t_features

    # def get_img_feature(self, image_id):
    #     """ decode the image feature """
    #     self.check_img_feature_file()
    #     self.check_img_feature_offset_map()

    #     if image_id in self.img_feat_offset_map:
    #         img_offset = self.img_feat_offset_map[image_id]
    #         self.img_feature_file.seek(img_offset, 0)
    #         arr = [s.strip() for s in self.img_feature_file.readline().split('\t')]
    #         num_boxes = int(arr[1])
    #         feat = np.frombuffer(base64.b64decode(arr[2]), dtype=np.float32).reshape((-1, self.args.img_feature_dim))
    #         return feat

    #     return None


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def score_with_cls_logits(logits, labels):
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



def trim_batch(batch):
    """ new batch func
    :param batch:
    :return:
    """
    print('batch size', len(batch))

    batch_size = len(batch)
    batch_tensors = []
    for ele in batch[0]:
        print(ele.shape, ele.size())
        zero_tensor = torch.zeros(([batch_size] + list(ele.size())))
        batch_tensors.append(zero_tensor)

    for b_id, b in enumerate(batch):
        print(b_id, len(b))
        for ele_id, ele in enumerate(b):
            print(ele_id, ele.shape)
            batch_tensors[ele_id][b_id] = ele
    return batch_tensors


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    #if args.local_rank in [-1, 0]: tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler, batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) # original

    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise NotImplementedError

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility (even between python 2 and 3)

    best_score = 0
    global_loss = 0.0
    global_score = 0.0
    best_model = {
        'epoch': 0,
        'model': copy.deepcopy(model), #model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    if args.freeze_steps >= 0:
        if hasattr(model, 'module'):
            model.module.freeze_backbone()
        else:
            model.freeze_backbone()
    #eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    for epoch in range(int(args.num_train_epochs)):
    #for epoch in train_iterator:
        #epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0

        if args.adjust_dp and epoch>=3:
            logger.info("change droput ratio {} to 0.3".format(args.drop_out))
            if hasattr(model, 'module'):
                model.module.dropout.p = 0.3
                model.module.bert.dropout.p = 0.3
                model.module.bert.embeddings.dropout.p = 0.3
            else:
                model.dropout.p = 0.3
                model.bert.dropout.p = 0.3
                model.bert.embeddings.dropout.p = 0.3

        if args.adjust_loss and epoch>=args.adjust_loss_epoch:
            logger.info("\t change loss type from kl to bce")
            model.loss_type = 'bce'

        # debug
        #epoch = 20
        #global_step = epoch*math.ceil(len(train_dataset)/(args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))

        t_start = time.time()
        for step, (indexes, batch) in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids_a':      batch[0],
                      'attention_mask_a': batch[1],
                      'token_type_ids_a': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'input_ids_b':      batch[3],
                      'attention_mask_b': batch[4],
                      'token_type_ids_b': batch[5],
                      'labels':         batch[6],
                      'img_feats':      None if args.img_feature_dim == -1 else batch[7]}
            outputs = model(**inputs)

            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss, logits = outputs[:2]

            #loss = instance_bce_with_logits(logits, batch[4])

            if args.n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                total_norm += torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                count_norm += 1

            global_loss += loss.item()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if global_step >= args.freeze_steps:
                    if hasattr(model, 'module'):
                        model.module.unfreeze_backbone()
                    else:
                        model.unfreeze_backbone()

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step)
                    )
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if args.local_rank not in [-1, 0]:
                        torch.distributed.barrier()

                    if args.local_rank in [-1, 0] and args.evaluate_during_training:
                    #if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score, upper_bound = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        logger.info("EVALERR: {}%".format(100 * best_score))

                    if args.local_rank == 0:
                        torch.distributed.barrier()

                    logging_loss = tr_loss

            #if args.max_steps > 0 and global_step > args.max_steps:
            #    epoch_iterator.close()
            #    break

        # evaluation
        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
        eval_result, eval_score, upper_bound = evaluate(args, model, eval_dataset, prefix=global_step)
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch>0 and epoch%args.save_epoch == 0) and (epoch>args.save_after_epoch):
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    logger.info("Saving model attempt: {}".format(save_num))
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)
        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))

        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch, t_end - t_start))

        #if args.max_steps > 0 and global_step > args.max_steps:
        #    train_iterator.close()
        #    break

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset=None, prefix=""):
    from seqeval.metrics import classification_report
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    #if args.n_gpu > 1: model = torch.nn.DataParallel(model) # debug: single-gpu or multi-gpus

    results = []
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        num_data = 0
        score = 0.0
        upper_bound = 0
        results_dict = {}
        y_true = []
        y_pred = []

        for indexes, batch in eval_dataloader:
        #for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids_a':      batch[0],
                        'attention_mask_a': batch[1],
                        'token_type_ids_a': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                        'input_ids_b':      batch[3],
                        'attention_mask_b': batch[4],
                        'token_type_ids_b': batch[5],
                        'img_feats':      None if args.img_feature_dim == -1 else batch[7]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                label_ids = batch[6].to('cpu').numpy()
                input_mask = batch[1].to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if ans2label_map[label_ids[i][j]] != "X" and ans2label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(ans2label_map[label_ids[i][j]])
                                temp_2.append(ans2label_map[logits[i][j]])
                        else:
                            #temp_1.pop()
                            #temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)

                # debug
                #val, idx = logits.max(1)
                #logger.info('idx: %s, batch[4]: %s' % (str(idx.shape), str(batch[3].shape)))
                #for i in range(idx.size(0)):
                #    logger.info('idx: %d, pred: %d, real: %d' % (idx[i].item(), eval_dataset.labels[idx[i].item()], batch[3][i].item()))

            nb_eval_steps += 1

            #if preds is None:
            #    preds = logits.detach().cpu().numpy()
            #    out_label_ids = inputs['labels'].detach().cpu().numpy()
            #else:
            #    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("***** Dev Eval results *****")
        logger.info("\n%s", report)
        #eval_true_label = np.concatenate(y_true)
        #eval_pred_label = np.concatenate(y_pred)
        #precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
        F_score_dev = float(report.split('\n')[-3].split('      ')[-2].split('    ')[-1])
        print("F-score: ", F_score_dev)
        # with open(os.path.join(args.data_dir, 'val_results.json'),
        #           'w') as f:
        #     json.dump(results_dict, f)

    t_end = time.time()
    logger.info('Eva Time Cost: %.3f' % (t_end - t_start))

        #eval_loss = eval_loss / nb_eval_steps
        #if args.output_mode == "classification":
        #    preds = np.argmax(preds, axis=1)
        #elif args.output_mode == "regression":
        #    preds = np.squeeze(preds)
        #result = compute_metrics(eval_task, preds, out_label_ids)
        #results.update(result)

        #output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        #with open(output_eval_file, "w") as writer:
        #    logger.info("***** Eval results {} *****".format(prefix))
        #    for key in sorted(result.keys()):
        #        logger.info("  %s = %s", key, str(result[key]))
        #        writer.write("%s = %s\n" % (key, str(result[key])))

    return results, score, upper_bound

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--txt_data_dir", default=None, type=str, required=True,
                        help="The input text data dir. Should contain the .json files (or other data files) for the task.")

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_file", type=str, default=None, help="Label Dictionary")
    parser.add_argument("--label2ans_file", type=str, default=None, help="Label to Answer Dictionary")

    parser.add_argument("--img_feat_dir", default=None, type=str, help="The input img_feat_dir.")
    parser.add_argument("--img_feat_format", default='pt', type=str, help="img_feat_format: pt or tsv.")

    parser.add_argument("--data_label_type", default='faster', type=str, help="faster or mask")
    parser.add_argument("--loss_type", default='kl', type=str, help="kl or xe")
    parser.add_argument("--use_vg", action='store_true', help="Use VG-QA or not.")
    parser.add_argument("--use_vg_dev", action='store_true', help="Use VG-QA as validation.")
    #parser.add_argument("--use_img_layernorm", action='store_true', help="use_img_layernorm")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train_val", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--do_test_dev", action='store_true', help="Whether to run test on the test-dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out for BERT.")
    parser.add_argument("--adjust_dp",action='store_true', help="Adjust Drop out for BERT.")

    parser.add_argument("--adjust_loss", action='store_true', help="Adjust Loss Type for BERT.")
    parser.add_argument("--adjust_loss_epoch", default=-1, type=int, help="Adjust Loss Type for BERT.")
    parser.add_argument("--classifier", default='linear', type=str, help="linear or mlp")
    parser.add_argument("--cls_hidden_scale", default=2, type=int, help="cls_hidden_scale: for classifier")

    parser.add_argument("--hard_label", action='store_true', help="Soft Label or Hard Label.")

    parser.add_argument("--max_img_seq_length", default=30, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='faster_r-cnn', type=str, help="faster_r-cnn or mask_r-cnn")
    parser.add_argument("--code_voc", default=512, type=int, help="dis_code_voc: 256, 512")
    parser.add_argument("--code_level", default='top', type=str, help="code level: top, botttom, both")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=5, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--add_od_labels', action='store_true')

    parser.add_argument("--philly", action='store_true', help="Use Philly: reset the output dir")
    parser.add_argument("--load_fast", action='store_true', help="Load Tensor Fast")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--max_phrases', type=int, default=5, help='maximal number of phrases in an example')
    parser.add_argument('--max_tag_length', type=int, default=20)
    parser.add_argument('--bivinvl_id2phrase', type=str, default=None)
    parser.add_argument('--use_single', action='store_true')
    parser.add_argument('--freeze_steps', type=int, default=-1)
    parser.add_argument('--mask_training', action='store_true')
    parser.add_argument('--use_b_text', action='store_true', help='whether to use tags in text')
    parser.add_argument('--use_b_feat', action='store_true', help='whether to use tags in feat')
    parser.add_argument('--dataset', type=str, default='twitter2015')

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 1 --img_feature_dim 565 --img_feature_type dis_code '

    #args = '--data_dir ../vqa/ban-vqa/data/qal_pairs --model_type bert --model_name_or_path bert-base-uncased --task_name vqa_text ' \
    #       '--do_train --do_eval --do_lower_case --max_seq_length 40 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --learning_rate 2e-5 ' \
    #       '--num_train_epochs 20.0 --output_dir ./results/vqa_text --label_file ../vqa/ban-vqa/data/cache/trainval_ans2label.pkl ' \
    #       '--save_steps 5000 --overwrite_output_dir --max_img_seq_length 10 --img_feature_dim 565 --img_feature_type other '

    #args = parser.parse_args(args.split())

    args = parser.parse_args()
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    # logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    #                     datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    args.output_mode = output_modes[args.task_name]
    num_labels = len(labels) + 1
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.code_voc = args.code_voc
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    #config.use_img_layernorm = args.use_img_layernorm

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #if args.do_eval:
    eval_dataset = NERDataset(args, 'test', tokenizer)

    if args.do_test:
        test_dataset = NERDataset(args, 'test', tokenizer)


    # Training
    if args.do_train:
        #train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        train_dataset = NERDataset(args, 'train', tokenizer)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`. They can then be reloaded using `from_pretrained()`
        #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        #model = model_class.from_pretrained(args.output_dir)
        #tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        #model.to(args.device)

    # Test
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, test_dataset, prefix=global_step)


if __name__ == "__main__":
    main()
