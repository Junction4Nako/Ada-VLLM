import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
import csv


class OscarTSVDataset(Dataset):
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True,
                 **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.num_readers = args.num_readers
        # self.corpus_tsvfile = TSVFile(os.path.join(self.root, self.cfg['corpus_file']))
        corpus_data = csv.reader(open(os.path.join(self.root, self.cfg['data_file']), 'r'))
        if 'textb_sample_mode' in kwargs:
            self.textb_sample_mode = kwargs['textb_sample_mode']
        else:
            self.textb_sample_mode = args.textb_sample_mode

        logging.info('Datasets: {}'.format(self.cfg['data_file']))
        self.num_splits = self.cfg['num_splits']
        self.image_faeture_path = [self.cfg['hdfs_root']+'split{}_obj_feats'.format(i+1) for i in range(self.num_splits)]
        self.image_label_path = [self.cfg['hdfs_root']+'split{}_od_tags'.format(i+1) for i in range(self.num_splits)]
        self.image_file_name = 'features.tsv'

        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        t_start = time.time()
        vid2split = json.load(open(os.path.join(self.root, self.cfg['vid2split']), 'r'))
        item2vid = json.load(open(os.path.join(self.root, self.cfg['item2vid']), 'r'))
        self.tags2id = json.load(open(os.path.join(self.root, self.cfg['tag2id']), 'r'))
        self.num_frames = self.cfg['num_frames']
        t_end = time.time()
        logging.info('Info: loading json mappings using {} secs'
                     .format(t_end - t_start))

        # load samples into memory
        if on_memory:
            self.all_docs = []
            self.all_qa_docs = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            max_tokens = 0
            next(corpus_data)
            for row in tqdm(corpus_data):
                doc = []
                item_id = row[0]
                text_a = row[1]
                if item_id not in item2vid:
                    continue

                # append id info
                vid = item2vid[item_id]
                if vid not in vid2split:
                    continue
                doc.append('%s|%s|%d' % (item_id, vid, vid2split[vid])) # item_id|vid|split
                # append text_a info
                self.corpus_lines = self.corpus_lines + 1
                sample = {"doc_id": len(self.all_docs), "line": len(doc)}
                self.sample_to_doc.append(sample)
                doc.append(text_a)
                # append text_b info
                self.corpus_lines = self.corpus_lines + 1
                tags = [t for t in row[-4:] if len(t)>0 and t!='NULL']
                tags = ';'.join([json.loads(t)['tag_name'] for t in tags])
                # tags = ''[strtags2id[t] for t in tags]
                doc.append(tags)

                # add to all_docs
                max_tokens = max(max_tokens, len(doc[1].split(' '))
                                 + len(doc[2].split(' ')))
                self.all_docs.append(doc)

            self.num_docs = len(self.all_docs)
            logging.info("Max_tokens: {}".format(max_tokens))
            del(corpus_data)
            del(item2vid)
            del(vid2split)
        # load samples later lazily from disk
        else:
            raise ValueError("on_memory = False Not supported yet!")

        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.corpus_lines - self.num_docs

    def get_img_info(self, idx):
        sample = self.sample_to_doc[idx]
        # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
        img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
        imgid2labels = self.imgid2labels[img_id]
        return {"height": imgid2labels["image_h"], "width": imgid2labels["image_w"]}

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                raise ValueError("on_memory = False Not supported yet!")

        vid, t1, t2, is_next_label, is_img_match, t3, split = self.random_sent(item)

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b1 = self.tokenizer.tokenize(t2[0])
            tokens_b2 = [self.tokenizer.tokenize(t) for t in t2[1]]
        else:
            tokens_b1 = None
            tokens_b2 = None

        # combine to one sample
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_c=t3,
                                   tokens_b1=tokens_b1, tokens_b2=tokens_b2, is_next=is_next_label,
                                   img_id=vid, is_img_match=is_img_match)

        # get image feature
        img_feat = self.get_img_feature(vid, split)
        tmp_max_img_seq_length = self.args.max_img_seq_length + self.num_frames*self.args.obj_per_frame
        if img_feat.shape[0] >= tmp_max_img_seq_length:
            img_feat = img_feat[0:tmp_max_img_seq_length, ]
            img_feat_len = img_feat.shape[0]
        else:
            img_feat_len = img_feat.shape[0]
            padding_matrix = torch.zeros((tmp_max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # transform sample to features
        cur_features = convert_example_to_features(self.args, cur_example,
                                                   self.seq_len, self.tokenizer,
                                                   img_feat_len, self.tags2id)

        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(cur_features.input_ids, dtype=torch.long),
                torch.tensor(cur_features.input_mask, dtype=torch.long),
                torch.tensor(cur_features.segment_ids, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
                torch.tensor(cur_features.is_next, dtype=torch.long),
                torch.tensor(cur_features.is_img_match, dtype=torch.long),
                item)
        else:
            return img_feat, (
                torch.tensor(cur_features.input_ids, dtype=torch.long),
                torch.tensor(cur_features.input_mask, dtype=torch.long),
                torch.tensor(cur_features.segment_ids, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
                torch.tensor(cur_features.is_next),
                torch.tensor(cur_features.is_img_match),
            ), item
        # return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        vid, t1, t2, t3, split = self.get_corpus_line(index)
        rand_dice = random.random()
        if rand_dice > 0.5:
            label = 0
            random_vid = vid
        elif rand_dice > self.args.texta_false_prob and t2 != "":
            # wrong qa triplets
            random_vid, t2 = self.get_random_line()
            label = 1
        else:
            # wrong retrieval triplets
            random_vid, t1 = self.get_random_texta()
            # args.num_contrast_classes = 3 if args.texta_false_prob<0.5 and (args.texta_false_prob>0 or not args.use_b) else 2
            label = self.args.num_contrast_classes-1

        img_match_label = 0
        if vid != random_vid: img_match_label = 1

        # assert len(t1) > 0
        # assert len(t2) > 0 or not self.args.use_b
        return vid, t1, t2, label, img_match_label, t3, split

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
            item_id, vid, split = self.all_docs[sample["doc_id"]][0].strip().split('|')
            split = int(split)
            t1 = self.all_docs[sample["doc_id"]][1]
            t2 = self.get_img_labels(vid, split)
            t3 = self.all_docs[sample["doc_id"]][2]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            self.current_vid = vid

            # assert t1 != ""
            if self.args.use_b or 'qa' in self.all_docs[sample["doc_id"]][0].split('_'):
                pass
                # assert t2 != ""
            else:
                t2 = ""
            return vid, t1, t2, t3, split
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        if self.on_memory:
            for _ in range(10):
                rand_doc_idx = random.randrange(0, len(self.all_docs))
                vid = self.all_docs[rand_doc_idx][0].split('|')[1]
                # check if our picked random line is really from another image like we want it to be
                if vid != self.current_vid:
                    break
            rand_doc = self.all_docs[rand_doc_idx]
            # img_id = rand_doc[0] # original
            item_id, vid, split = rand_doc[0].split('|')
            split = int(split)
            line = self.get_img_labels(vid, split)
            return vid, line
        else:
            raise ValueError("on_memory = False Not supported yet!")

    def get_random_texta(self):
        """
        Get random text_a from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        if self.on_memory:
            for _ in range(10):
                rand_doc_idx = random.randrange(0, len(self.all_docs))
                vid = self.all_docs[rand_doc_idx][0].split('|')[1]
                # check if our picked random line is really from another image like we want it to be
                if vid != self.current_vid:
                    break
            rand_doc = self.all_docs[rand_doc_idx]
            # img_id = rand_doc[0] # original
            vid = rand_doc[0].split('|')[1]
            line = rand_doc[1] # we want the text_a
            return vid, line
        else:
            raise ValueError("on_memory = False Not supported yet!")


    def get_img_labels(self, vid, split=0):
        input_keys = [vid+'.jpg'] + [vid + '_{}.jpg'.format(i) for i in range(self.num_frames)]
        frame_tags = []
        cov_tags = None
        for i,k in enumerate(input_keys):
            try:
                v = self.od_reader[split].read_many([k])[0]
            except:
                v = self.od_reader[split+1].read_many([k])[0]
            tmp_tags = str(v, encoding='utf-8').split()
            if i == 0:
                if len(tmp_tags) > self.args.max_img_seq_length:
                    tmp_tags = tmp_tags[:self.args.max_img_seq_length]
                cov_tags = ' '.join(tmp_tags)
            else:
                if len(tmp_tags) > self.args.tag_per_frame:
                    tmp_tags = tmp_tags[:self.args.tag_per_frame]
                frame_tags.append(' '.join(tmp_tags))
                # tokenized_result = self.tokenizer.tokenize(' '.join(tmp_tags))
                # if len(tokenized_result) > self.args.tag_per_frame:
                #     print(tokenized_result)
                #     print(tmp_tags)
                #     raise ValueError
        return cov_tags, frame_tags


    def get_img_feature(self, vid, split=0):
        """ decode the image feature: read the image feature from the right chunk id """
        input_keys = [vid+'.jpg'] + [vid + '_{}.jpg'.format(i) for i in range(self.num_frames)]
        full_img_feats = []
        for i, k in enumerate(input_keys):
            try:
                v = self.img_reader[split].read_many([k])[0]
            except:
                v = self.img_reader[split+1].read_many([k])[0]
            t_features = np.frombuffer(v, dtype=np.float32).reshape(-1, self.args.img_feature_dim)
            if not t_features.flags['WRITEABLE']:
                new_features = np.copy(t_features)
            else:
                new_features = t_features
            if i == 0:
                # cover image
                if new_features.shape[0] > self.args.max_img_seq_length:
                    new_features = new_features[:self.args.max_img_seq_length, :]
            else:
                # frame features
                if new_features.shape[0] > self.args.obj_per_frame and self.args.visual_random:
                    tmp_features = np.zeros((self.args.obj_per_frame, new_features.shape[1]))
                    for i in range(self.args.obj_per_frame):
                        prob = random.random()
                        if prob < 0.15:
                            prob /= 0.15
                            if prob < 0.5:
                                continue
                            else:
                                r_idx = random.randint(self.args.obj_per_frame, new_features.shape[0]-1)
                                tmp_features[i, :] = new_features[r_idx, :]
                        else:
                            tmp_features[i, :] = new_features[i, :]
                else:
                    tmp_features = new_features[:self.args.obj_per_frame, :]

            new_features = torch.tensor(new_features, dtype=torch.float)
            full_img_feats.append(new_features)
        return torch.cat(full_img_feats, dim=0)


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b1=None, tokens_b2=None, is_next=None,
                 lm_labels=None, img_id=None, is_img_match=None, tokens_c=None,
                 img_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b1 = tokens_b1
        self.tokens_b2 = tokens_b2
        self.tokens_c = tokens_c
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

        self.img_id = img_id
        self.is_img_match = is_img_match
        self.img_label = img_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next,
                 lm_label_ids, img_feat_len, is_img_match):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logging.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(
                        token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_cate(tokens, tokenizer, phrase2id):
    output_labels = []
    tokens = [phrase2id[t] for t in tokens if t in phrase2id]
    flag = False
    b_s = min(phrase2id.values())
    for i, t in enumerate(tokens):
        if flag:
            replace = True
            prob = random.random()
        else:
            prob = random.random()
            if prob < 0.15:
                replace = True
                flag = True
                prob /= 0.15
            else:
                replace = False
        if replace:
            if prob < 0.8:
                tokens[i] = tokenizer.vocab['[MASK]']
            elif prob < 0.9:
                tokens[i] = random.randint(0, len(phrase2id)-1) + b_s
            output_labels.append(t)
        else:
            output_labels.append(-1)
    return tokens, output_labels
                


def convert_example_to_features(args, example, max_seq_length, tokenizer,
                                img_feat_len, phrase2id):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = example.tokens_a
    tokens_b1 = example.tokens_b1
    tokens_b2 = example.tokens_b2
    tokens_c = example.tokens_c.split(';')
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b1, max_seq_length - 3)

    is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type == 1:
        is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    # if not args.mask_loss_for_unmatched and is_next_type == 2:
    #     t1_label = [-1]*len(tokens_a)
    # else:
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    tokens_b = tokens_b1
    for tmp in tokens_b2:
        if len(tmp)>args.tag_per_frame:
            tmp = tmp[:args.tag_per_frame]
        tokens_b.extend(tmp)
    if tokens_b:
        if not args.mask_loss_for_unmatched and is_next_type == 1:
            t2_label = [-1]*len(tokens_b)
        else:
            tokens_b, t2_label = random_word(tokens_b, tokenizer)
        
    tokens_c, t3_label = random_cate(tokens_c, tokenizer, phrase2id)

    # concatenate lm labels and account for CLS, SEP, SEP
    if tokens_b:
        lm_label_ids = ([-1] + t1_label + [-1] + t2_label + t3_label + [-1])
    else:
        lm_label_ids = ([-1] + t1_label + t3_label + [-1])

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    input_ids = input_ids + tokens_c
    segment_ids = segment_ids + [1] * len(tokens_c)

    input_ids.append(tokenizer.vocab["[SEP]"])
    segment_ids.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    tmp_max_seq_length = max_seq_length + args.num_frames*args.tag_per_frame + 4
    while len(input_ids) < tmp_max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    # print(len(tokens_a), len(tokens_b1), ';'.join([str(len(t)) for t in tokens_b2]), len(tokens_c))
    assert len(input_ids) == tmp_max_seq_length, "input_ids length: {} exceeds limit {}".format(len(input_ids), tmp_max_seq_length)
    assert len(input_mask) == tmp_max_seq_length, "input_mask length: {} exceeds limit {}".format(len(input_mask), tmp_max_seq_length)
    assert len(segment_ids) == tmp_max_seq_length, "segment_ids length: {} exceeds limit {}".format(len(segment_ids), tmp_max_seq_length)
    assert len(lm_label_ids) == tmp_max_seq_length, "lm_label_ids length: {} exceeds limit {}".format(len(lm_label_ids), tmp_max_seq_length)

    # image features
    tmp_max_img_seq_length = args.max_img_seq_length + args.num_frames*args.obj_per_frame
    if img_feat_len > tmp_max_img_seq_length:
        input_mask = input_mask + [1] * img_feat_len
    else:
        input_mask = input_mask + [1] * img_feat_len
        pad_img_feat_len = tmp_max_img_seq_length - img_feat_len
        input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * tmp_max_img_seq_length

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("LM label: %s " % lm_label_ids)
        logging.info("Is next sentence label: %s " % example.is_next)

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             img_feat_len=img_feat_len,
                             is_img_match=example.is_img_match)
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()