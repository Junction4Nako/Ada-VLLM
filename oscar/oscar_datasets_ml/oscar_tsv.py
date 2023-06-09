import os
import time
import json
import logging
import random
import glob
import base64
# from datasets.fingerprint import get_datasets_with_cache_file_in_temp_dir
from torch.nn.functional import GRID_SAMPLE_PADDING_MODES
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
from collections import Counter

phrase_vocab_size = None
theme_vocab_size = None

class OscarTSVDataset_C(Dataset):
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True, ds_names=None,
                 **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_tsvfile = TSVFile(os.path.join(self.root, self.cfg['corpus_file']), generate_lineidx=True)
        if 'textb_sample_mode' in kwargs:
            self.textb_sample_mode = kwargs['textb_sample_mode']
        else:
            self.textb_sample_mode = args.textb_sample_mode

        if ds_names is None:
            self.only_image = False
            self.datasets_names = self.cfg['corpus'].split('_')
        else:
            self.only_image = True
            self.datasets_names = ds_names.split('_')
        self.datasets_with_splits = ['googlecc', 'sbu', 'oi', 'objects365', 'tagoi']
        self.datasets_with_onesplit = ['coco', 'flickr30k', 'gqa']
        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_label_path = self.cfg['image_label_path']
        fk_test_ids = json.load(open(os.path.join(self.root, self.cfg['fk_test_id']), 'r'))
        for key, val in self.image_label_path.items():
            # get the absolute path
            if key in self.datasets_names:
                self.image_label_path[key] = os.path.join(self.root, val)
        self.image_feature_path = self.cfg['image_feature_path']
        self.image_file_name = 'features.tsv'
        if args.data_dir is not None:
            for key, val in self.image_feature_path.items():
                # get the absolute path
                if key in self.datasets_names:
                    self.image_feature_path[key] = os.path.join(args.data_dir,
                                                                val)
                else:
                    logging.info("Data {} with path {} is not used in the "
                                 "training.".format(key, val))
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

        # loading text 2 concepts
        logging.info('Loading text phrases from: {}'.format(self.cfg['sg_file']))
        concept_list = json.load(open(os.path.join(self.root, self.cfg['sg_file']), 'r'))
        qa_ans_mapping = json.load(open(os.path.join(self.root, self.cfg['qa_ans2label']), 'r'))
        # id2phrase = json.load(open(os.path.join(self.root, self.cfg['id2sg_file']), 'r'))
        # phrase2id = {tuple(v):int(k) for k,v in id2phrase.items()}
        # bad_cases = json.load(open(os.path.join(self.root, self.cfg['bad_cases']), 'r'))
        global phrase_vocab_size
        phrase_vocab_size = self.cfg['phrase_vocab_size']
        self.num_phrases = args.max_phrases
        
        # load img 2 theme concept
        # logging.info('Loading visual theme from: {}'.format(self.cfg['theme_json']))
        # self.img2theme = json.load(open(os.path.join(self.root, self.cfg['theme_json']), 'r'))
        # self.img2theme = {k:[s[0] for s in v] for k,v in self.img2theme.items()}
        # global theme_vocab_size
        # theme_vocab_size = self.cfg['theme_vocab_size']
        self.num_themes = args.max_visual_themes

        self.chunk_list = None
        if 0 <= args.chunk_start_id <= args.chunk_end_id and args.chunk_end_id >= 0:
            self.chunk_list = [str(c_i) for c_i in range(args.chunk_start_id,
                                                    args.chunk_end_id)]
            logging.info('Chunk list: {}'.format(','.join(self.chunk_list)))

        # load image tags and features
        t_start = time.time()
        self.img_label_file = None
        self.img_qa_file = None
        self.img_label_offset_map = None
        self.img_qa_offset_map = None
        self.img_feature_file = None
        self.img_feat_offset_map = None
        self.load_img_labels()
        self.load_img_tsv_features()
        # print(self.img_feat_offset_map.keys())
        # for k in self.img_feat_offset_map.keys():
        #     if k in self.datasets_with_splits:
        #         print(k, self.img_feat_offset_map[k].keys())
        t_end = time.time()
        logging.info('Info: loading img features using {} secs'
                     .format(t_end - t_start))
        fk_count = 0

        # load samples into memory
        if on_memory:
            self.all_docs = []
            self.tuple_mask_map = []
            self.all_qa_docs = []
            self.all_qa_ans = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            max_tokens = 0
            # tmp_tag2id = Counter()
            for line_no in tqdm(range(len(self.corpus_tsvfile))):
            # for line_no in tqdm(range(10000)):
                doc = []
                row = self.corpus_tsvfile.seek(line_no)
                img_info = row[0].split('_')
                label_info = row[1].split('_')
                if 'qa' in label_info and args.only_cap:
                    # only use caption pair since Q-A pair is more complicated
                    continue
                if self.args.only_qa and 'qa' not in label_info:
                    continue
                if self.only_image and 'qa' in label_info:
                    # when creating image only dataset, no QA Pairs are considered
                    continue
                assert img_info[0] == label_info[
                    0], "Dataset names for image and label do not match!"
                dataset_name = label_info[0]
                if dataset_name == 'flickr30k' and args.no_fk:
                    if int(img_info[-1]) in fk_test_ids:
                        # print(123)
                        if fk_count == 0:
                            logging.info('found flickr test id in pretrain corpus')
                        fk_count += 1
                        continue
                if dataset_name == 'cc':
                    dataset_name = 'googlecc'

                if dataset_name not in self.datasets_names:
                    continue

                if dataset_name in self.datasets_with_splits:
                    chunk_id = img_info[-2]
                    if self.chunk_list is not None and chunk_id not in self.chunk_list:
                        continue
                    else:
                        img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
                else:
                    img_feat_offset_map = self.img_feat_offset_map[dataset_name]
                assert img_info[-1] in img_feat_offset_map, "{}: Image id {} cannot be found in image feature imageid_to_index file!".format(row[0], img_info[-1])

                # append id info
                doc.append('%s|%s' % (row[0], row[1]))
                # append text_a info
                self.corpus_lines = self.corpus_lines + 1
                sample = {"doc_id": len(self.all_docs), "line": len(doc)}
                self.sample_to_doc.append(sample)
                assert len(row[2]) != 0, "Text_a is empty in {} : {}"\
                    .format(dataset_name, row[0])
                doc.append(row[2])
                # append text_b info
                self.corpus_lines = self.corpus_lines + 1
                label_id = label_info[-1]
                if 'qa' in label_info:
                    assert img_info[-1] == label_info[
                        -2], "Image ids for image and qa do not match!"
                    label_line_no = self.img_qa_offset_map[dataset_name][label_id]
                    rowb = self.img_qa_file[dataset_name].seek(label_line_no)
                    # also reach the image object tags
                    # img_id = label_info[-2]
                    # add_line_no = self.img_label_offset_map[dataset_name][img_id]
                    # add_rowb = self.img_label_file[dataset_name]
                else:
                    assert img_info[-1] == label_info[
                        -1], "Image ids for image and label do not match!"
                    label_line_no = self.img_label_offset_map[dataset_name][label_id]
                    rowb = self.img_label_file[dataset_name].seek(label_line_no)
                assert label_id == rowb[0]
                results = json.loads(rowb[1])
                if 'qa' not in label_info: # more intuitively, should be if 'qa' not in label_info:
                    objects = results['objects']
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": results["image_h"], "image_w": results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert results["image_h"] == self.imgid2labels[row[0]][
                            "image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert results["image_w"] == self.imgid2labels[row[0]][
                            "image_w"], "Image_w does not match in image {}!".format(row[0])
                    if args.use_gtlabels and 'gt_objects' in results:
                        # use ground-truth tags for text_b
                        tmp = list(set([cur_d['class'] for cur_d in results["gt_objects"]])) # use set here to avoid duplication
                        textb = '\t'.join(tmp) # change from ' ' to '\t' here
                    else:
                        tmp = list(set([cur_d['class'] for cur_d in objects])) # use set here to avoid duplication
                        textb = '\t'.join(tmp) # change from ' ' to '\t' here
                    add_textb = None
                    qa_textb = -1
                    # tmp_tag2id.update(tmp)
                else:
                    tag_label_line_no = self.img_label_offset_map[dataset_name][img_info[-1]]
                    tag_rowb = self.img_label_file[dataset_name].seek(tag_label_line_no)
                    tag_results = json.loads(tag_rowb[1])
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": tag_results["image_h"], "image_w": tag_results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert tag_results["image_h"] == self.imgid2labels[row[0]][
                            "image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert tag_results["image_w"] == self.imgid2labels[row[0]][
                            "image_w"], "Image_w does not match in image {}!".format(row[0])
                    # qa_textb = ' '.join(results['labels'])
                    # try to add QA supervision into pretraining
                    if 'conf' in results:
                        # VQA pairs
                        qa_textb = [(l, results['conf'][i])for i,l in enumerate(results['labels']) if l in qa_ans_mapping]
                        if len(qa_textb) == 0:
                            qa_textb = -1
                        else:
                            qa_textb = qa_ans_mapping[sorted(qa_textb, key=lambda x:-1*x[1])[0][0]]
                    else:
                        # GQA pairs
                        qa_textb = results['labels'][0] # only one answer available
                        qa_textb = qa_ans_mapping[qa_textb] if qa_textb in qa_ans_mapping else -1
                    # add_results = json.loads(add_rowb[1])
                    if args.use_gtlabels and 'gt_objects' in results:
                        # use ground-truth tags for text_b
                        tmp = list(set([cur_d['class'] for cur_d in tag_results["gt_objects"]])) # # use set here to avoid duplication
                        add_textb = '\t'.join(tmp) # change from ' ' to '\t' here
                    else:
                        tmp = list(set([cur_d['class'] for cur_d in tag_results["objects"]]))
                        add_textb = '\t'.join(tmp) # change from ' ' to '\t' here
                    textb = add_textb
                    # tmp_tag2id.update(tmp)
                assert len(textb) != 0, "Text_b is empty in {} : {}".format(dataset_name, row[1])
                doc.append(textb)
                # if add_textb is not None:
                #     doc.append(add_textb)

                # add to all_docs
                max_tokens = max(max_tokens, len(doc[1].split(' '))
                                 + len(doc[2].split(' ')))
                if 'qa' in label_info:
                    self.all_qa_ans.append(qa_textb)
                    # self.all_qa_docs.append({"doc":doc, "doc_id": len(self.all_docs)})
                else:
                    self.all_qa_ans.append(qa_textb)
                
                # add extra concepts
                # concepts[0] = [concept_0, concept_1, ..., concept_n] (all in token ids)
                # concepts[1] = {word_index:[concept_index]}, indicate the concept-word relation
                concepts = concept_list[line_no]
                # current_con = []
                # old2new = {}
                # for c_ind, cc in enumerate(concepts[0]):
                #     c_phrase_id = phrase2id[tuple(cc)]
                #     if c_phrase_id not in bad_cases:
                #         old2new[c_ind] = len(current_con)
                #         current_con.append(c_phrase_id)
                c_mask = {int(k):v for k,v in concepts[1].items()}
                # c_mask = {}
                # for k,v in concepts[1].items():
                #     tmp_k = int(k)
                #     tmp_v = [old2new[kv] for kv in v if kv in old2new]
                #     if len(tmp_v) > 0:
                #         c_mask[tmp_k] = tmp_v
                assert concepts[2]==row[1]
                # doc.append(current_con)
                doc.append(concepts[0])
                self.tuple_mask_map.append(c_mask)
                
                self.all_docs.append(doc)
                # if len(self.all_docs) > 100000:
                #     break

            self.num_docs = len(self.all_docs)
            logging.info("Max_tokens: {}".format(max_tokens))
        # load samples later lazily from disk
        else:
            raise ValueError("on_memory = False Not supported yet!")

        logging.info(
            "deleted {} lines from pretrain corpus from flickr test/val".format(fk_count)
        )
        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )
        # tmp_to_save = {}
        # i = 0
        # for k in tmp_tag2id.keys():
        #     tmp_to_save[k] = i
        #     i += 1
        # with open(os.path.join(self.root, 'tag2id.json'), 'w') as wf:
        #     json.dump(tmp_to_save, wf)
        assert len(self.tuple_mask_map) == len(self.all_docs)
        del(concept_list)
        del(qa_ans_mapping)
        # del()

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.corpus_lines - self.num_docs

    def get_img_info(self, idx):
        sample = self.sample_to_doc[idx]
        # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
        img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
        imgid2labels = self.imgid2labels[img_id]
        return {"height": imgid2labels["image_h"], "width": imgid2labels["image_w"]}

    # @profile(precision=4,stream=open('memory_profiler.log','w+'))
    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                raise ValueError("on_memory = False Not supported yet!")

        img_id, t1, t2, is_next_label, is_img_match, qa_ans, p_c, doc_idx = self.random_sent(item)
        phrase_mask_map = self.tuple_mask_map[doc_idx]

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None

        # combine to one sample
        if self.args.no_phrase:
            p_c = []
        cur_example = InputExample(guid=cur_id, tokens_a=tokens_a,
                                   tokens_b=tokens_b, is_next=is_next_label,
                                   img_id=img_id, is_img_match=is_img_match,
                                   qa_ans=qa_ans, phrase_concept=p_c,
                                   phrase_mask_map=phrase_mask_map)

        # get image feature
        img_feat = self.get_img_feature(img_id)
        if img_feat.shape[0] >= self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            img_feat_len = img_feat.shape[0]
        
        if self.args.visual_learning:
            target_img_feat = img_feat.clone()
            tags = self.sample_to_doc[item][-1]
            img_feat, visual_labels, mask_region_id = random_visual(img_feat, tags, self.args.tag2id)


        if img_feat.shape[0] < self.args.max_img_seq_length:
            img_feat_len = img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]), dtype=self.args.dtype)
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.visual_learning:
                target_img_feat = torch.cat((target_img_feat, padding_matrix), dim=0)
                visual_labels += [-1]*(self.args.max_img_seq_length - img_feat_len)
                mask_region_id += [0]*(self.args.max_img_seq_length - img_feat_len)

        # transform sample to features
        cur_features = convert_example_to_features(self.args, cur_example,
                                                   self.seq_len, self.tokenizer,
                                                   img_feat_len, self.num_phrases, self.num_themes)

        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(cur_features.input_ids_a, dtype=torch.long),
                torch.tensor(cur_features.input_mask_a, dtype=torch.long),
                torch.tensor(cur_features.segment_ids_a, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids_a, dtype=torch.long),
                torch.tensor(cur_features.input_ids_b, dtype=torch.long),
                torch.tensor(cur_features.input_mask_b, dtype=torch.long),
                torch.tensor(cur_features.segment_ids_b, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids_b, dtype=torch.long),
                torch.tensor(cur_features.is_next, dtype=torch.long),
                torch.tensor(cur_features.is_img_match, dtype=torch.long),
                torch.tensor(cur_features.phrase_index, dtype=torch.long),
                torch.tensor(cur_features.image_index, dtype=torch.long),
                torch.tensor(cur_features.qa_ans, dtype=torch.long),
                item)
        else:
            return img_feat, (
                torch.tensor(cur_features.input_ids, dtype=torch.long),
                torch.tensor(cur_features.input_mask, dtype=torch.long),
                torch.tensor(cur_features.segment_ids, dtype=torch.long),
                torch.tensor(cur_features.lm_label_ids, dtype=torch.long),
                torch.tensor(cur_features.is_next),
                torch.tensor(cur_features.is_img_match)
                ), item
        # return cur_tensors

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        img_id, t1, t2, qa_ans, p_c = self.get_corpus_line(index)
        # qa_ans = None if not a QA-pair
        doc_idx = index
        rand_dice = random.random()
        if rand_dice >= 0: # changed to >=0 here to make it always true
            label = 0
            random_img_id = img_id
        elif rand_dice > self.args.texta_false_prob and t2 != "":
            # wrong qa triplets
            random_img_id, t2, n_v_c = self.get_random_line()
            if self.args.change_theme:
                v_c = n_v_c
            label = 1
        else:
            # wrong retrieval triplets
            random_img_id, t1, p_c, doc_idx = self.get_random_texta()
            # args.num_contrast_classes = 3 if args.texta_false_prob<0.5 and (args.texta_false_prob>0 or not args.use_b) else 2
            label = self.args.num_contrast_classes-1

        img_match_label = 0
        if img_id != random_img_id: img_match_label = 1

        assert len(t1) > 0
        assert len(t2) > 0 or not self.args.use_b
        return img_id, t1, t2, label, img_match_label, qa_ans, p_c, doc_idx

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
            img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
            only_img_id = img_id.split('_')
            only_img_id = only_img_id[0]+'_'+only_img_id[-1]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            self.current_img = img_id

            # get extra concepts
            # v_c = self.img2theme[only_img_id] # visual theme concepts
            qa_ans = self.all_qa_ans[item]
            p_c = self.all_docs[sample["doc_id"]][-1] # textual phrase concepts

            assert t1 != ""
            if self.args.use_b or 'qa' in self.all_docs[sample["doc_id"]][1].split('_'):
                assert t2 != ""
            else:
                t2 = ""
            return img_id, t1, t2, qa_ans, p_c
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
            if self.textb_sample_mode in [0, 1]:
                # sample from all docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_docs))
                    img_id = self.all_docs[rand_doc_idx][0].split('|')[0]
                    # check if our picked random line is really from another image like we want it to be
                    if img_id != self.current_img:
                        break
                rand_doc = self.all_docs[rand_doc_idx]
            else:
                # sample from all qa docs
                for _ in range(10):
                    rand_doc_idx = random.randrange(0, len(self.all_qa_docs))
                    # check if our picked random line is really from another doc like we want it to be % no need to be different image here
                    if self.all_qa_docs[rand_doc_idx]["doc_id"] != self.current_doc:
                        break
                rand_doc = self.all_qa_docs[rand_doc_idx]["doc"]
            # img_id = rand_doc[0] # original
            img_id = rand_doc[0].split('|')[0]
            if self.textb_sample_mode == 0:
                # default oscar sample mode
                line = rand_doc[random.randrange(1, len(rand_doc))]
            else:
                # only sample text_b
                line = rand_doc[2]
            only_img_id = img_id.split('_')
            only_img_id = only_img_id[0]+'_'+only_img_id[-1]
            v_c = self.img2theme[only_img_id]
            return img_id, line, v_c
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
                img_id = self.all_docs[rand_doc_idx][0].split('|')[0]
                # check if our picked random line is really from another image like we want it to be
                if img_id != self.current_img:
                    break
            rand_doc = self.all_docs[rand_doc_idx]
            # img_id = rand_doc[0] # original
            img_id = rand_doc[0].split('|')[0]
            line = rand_doc[1] # we want the text_a
            p_c = rand_doc[-1] # text_a with its phrase concepts
            return img_id, line, p_c, rand_doc_idx
        else:
            raise ValueError("on_memory = False Not supported yet!")

    # tsv image labels
    def load_img_labels(self):
        self.check_img_label_file()
        self.check_img_label_offset_map()

    def check_img_label_file(self):
        if self.img_label_file is None:
            self.img_label_file = {}
            self.img_qa_file = {}
            for dataset_name in self.datasets_names:
                img_label_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'predictions_gt.tsv')
                img_qa_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_fileB.tsv')
                t_s = time.time()
                self.img_label_file[dataset_name] = TSVFile(img_label_file_path)
                if os.path.exists(img_qa_file_path):
                    self.img_qa_file[dataset_name] = TSVFile(img_qa_file_path)
                t_e = time.time()
                logging.info(
                    "Open image label file {}, time: {}".format(
                        img_label_file_path, (t_e - t_s)))

    def check_img_label_offset_map(self):
        if self.img_label_offset_map is None:
            self.img_label_offset_map = {}
            self.img_qa_offset_map = {}
            for dataset_name in self.datasets_names:
                img_label_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'imageid2idx.json')
                img_qa_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_qaid2idx.json')
                t_s = time.time()
                self.img_label_offset_map[dataset_name] = json.load(
                    open(img_label_offset_map_path))
                if os.path.exists(img_qa_offset_map_path):
                    self.img_qa_offset_map[dataset_name] = json.load(
                        open(img_qa_offset_map_path))
                t_e = time.time()
                logging.info(
                    "Load img label offset map: {}, time: {}".format(
                        img_label_offset_map_path, (t_e - t_s)))

    def get_img_labels(self, image_id):
        """ decode the image labels: read the image label from the img_label.tsv """
        self.check_img_label_file()
        self.check_img_label_offset_map()

        if image_id in self.img_label_offset_map:
            img_offset = self.img_label_offset_map[image_id]

            self.img_label_file.seek(img_offset, 0)
            arr = [s.strip() for s in
                   self.img_label_file.readline().split('\t')]
            eles = json.loads(arr[1])
            labels = eles['labels']
            return labels

        return None

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            # self.img_feature_file = [] # original
            self.img_feature_file = {}
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading dataset {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    self.img_feature_file[dataset_name] = {}
                    self.img_feat_offset_map[dataset_name] = {}
                    chunk_list = []
                    if self.chunk_list is not None:
                        chunk_list = self.chunk_list
                        chunk_file_list = []
                        for chunk_fp_id in chunk_list:
                            chunk_file_list.append(
                                os.path.join(self.image_feature_path[dataset_name], chunk_fp_id, self.image_file_name)
                            )
                        if dataset_name == 'googlecc':
                            for i, (chunk_fp_id, chunk_fp) in enumerate(zip(chunk_list, chunk_file_list)):
                                assert os.path.exists(chunk_file_list[i]), "Chunk file {} does not exists!".format(chunk_fp)
                    else:
                        chunk_file_list = glob.glob(
                            self.image_feature_path[dataset_name] + "/*/{}".format(self.image_file_name)
                        )
                        for chunk_fp in chunk_file_list:
                            chunk_fp_id = chunk_fp.split('/')[-2]
                            chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feature_file[dataset_name][chunk_fp_id] = TSVFile(chunk_fp)
                        chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                        assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_offsetmap, 'r'))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                    logging.info(
                        "Image chunk info: {}".format('\n'.join(chunk_file_list))
                    )
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = os.path.join(self.image_feature_path[dataset_name], self.image_file_name)
                    self.img_feature_file[dataset_name] = TSVFile(chunk_fp)
                    chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                    assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                    self.img_feat_offset_map[dataset_name] = json.load(open(chunk_offsetmap, 'r'))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError("Not supported dataset: {}".format(dataset_name))

    def check_img_feature_offset_map(self):
        """ load the image feature offset map """
        if self.img_feat_offset_map is None:
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading imageid2idx_map {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    chunk_list = []
                    chunk_file_list = glob.glob(
                        self.image_feature_path[
                            dataset_name] + "/*/imageid2idx.json"
                    )
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_fp))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = self.image_feature_path[
                                   dataset_name] + "/imageid2idx.json"
                    self.img_feat_offset_map[dataset_name] = json.load(
                        open(chunk_fp))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError(
                        "Not supported dataset: {}".format(dataset_name))

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        self.check_img_feature_file()
        self.check_img_feature_offset_map()
        img_infos = image_id.split('_')
        dataset_name = img_infos[0]
        if dataset_name == 'cc':
            dataset_name = 'googlecc'
        img_id = img_infos[-1]
        if dataset_name in self.datasets_with_splits:
            chunk_id = img_infos[-2]
            img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
            img_feature_file = self.img_feature_file[dataset_name][chunk_id]
        else:
            img_feat_offset_map = self.img_feat_offset_map[dataset_name]
            img_feature_file = self.img_feature_file[dataset_name]
        if img_id in img_feat_offset_map:
            img_offset = img_feat_offset_map[img_id]

            arr = img_feature_file.seek(img_offset)
            num_boxes = int(arr[1])
            feat = np.frombuffer(base64.b64decode(arr[-1]),
                                 dtype=np.float32).reshape(
                (num_boxes, self.args.img_feature_dim))
            if not feat.flags['WRITEABLE']:
                feat = np.copy(feat)
            # feat = torch.from_numpy(feat)
            feat = torch.tensor(feat, dtype=self.args.dtype)
            return feat

        return None


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None,
                 lm_labels=None, img_id=None, is_img_match=None,
                 img_label=None, qa_ans=None, phrase_concept=None,
                 phrase_mask_map=None):
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
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

        self.img_id = img_id
        self.is_img_match = is_img_match
        self.img_label = img_label
        self.qa_ans = qa_ans
        self.phrase_concept = phrase_concept
        self.phrase_mask_map = phrase_mask_map


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_a, input_mask_a, segment_ids_a, is_next,
                 lm_label_ids_a, img_feat_len, is_img_match, phrases_index, image_index,
                 input_ids_b, input_mask_b, segment_ids_b, lm_label_ids_b, qa_ans=None):
        self.input_ids_a = input_ids_a
        self.input_mask_a = input_mask_a
        self.segment_ids_a = segment_ids_a
        self.is_next = is_next
        self.lm_label_ids_a = lm_label_ids_a
        self.qa_ans = qa_ans

        self.input_ids_b = input_ids_b
        self.input_mask_b = input_mask_b
        self.segment_ids_b = segment_ids_b
        self.lm_label_ids_b = lm_label_ids_b

        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match
        self.phrase_index = phrases_index
        self.image_index = image_index


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

def random_phrases(tokenizer, phrase_nodes, t1_label, phrase_mask_map):
    output_label = []
    already_masked = set()
    for i,t in enumerate(t1_label):
        if t >= 0:
            if i in phrase_mask_map:
                already_masked.update(phrase_mask_map[i])
    # print('test:', [i for i,t in enumerate(t1_label) if t>=0], phrase_mask_map, already_masked)
    for i, phrase in enumerate(phrase_nodes):
        if i in already_masked:
            output_label.append(phrase)
            phrase_nodes[i] = tokenizer.vocab['[MASK]']
        else:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    phrase_nodes[i] = tokenizer.vocab['[MASK]']

                # 10% randomly change token to random token
                elif prob < 0.9:
                    phrase_nodes[i] = random.randint(0, phrase_vocab_size-1)+tokenizer.vocab_size
                output_label.append(phrase)
            else:
                output_label.append(-1)
    return phrase_nodes, output_label


def random_theme(theme_nodes, tokenizer):
    output_label = []
    for i, t in enumerate(theme_nodes):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.8:
                theme_nodes[i] = tokenizer.vocab['[MASK]'] - len(tokenizer.vocab) - phrase_vocab_size

            elif prob < 0.9:
                theme_nodes[i] = random.randint(0, theme_vocab_size-1)
            output_label.append(t)
        else:
            output_label.append(-1 - len(tokenizer.vocab) - phrase_vocab_size)
    return theme_nodes, output_label


def random_visual(regions, od_tags, tag2id):
    """
    Masking some random regions for Masked Region task with probabilities as in the VLP papers.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    od_labels = od_tags.lower().split('\t')
    output_label = []
    mask_region_id = []

    # print(od_labels, len(od_labels), regions.shape[0])
    for i in range(regions.shape[0]):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            regions[i, :-6] = 0 # mask region
            output_label.append(tag2id[od_labels[i]] if od_labels[i] in tag2id else -1)
            mask_region_id.append(1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            mask_region_id.append(0)

    return regions, output_label, mask_region_id

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def convert_example_to_features(args, example, max_seq_length, tokenizer,
                                img_feat_len, num_phrases, num_themes):
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

    # extra_concept part
    qa_ans = example.qa_ans
    phrase_nodes = example.phrase_concept
    phrase_mask_map = example.phrase_mask_map
    
    tokens_a = example.tokens_a
    tokens_b = None
    if example.tokens_b:
        tokens_b = example.tokens_b
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        if len(tokens_b) > args.max_tag_length - 2:
            tokens_b = tokens_b[:(args.max_tag_length)-2]
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # is_next_type = example.is_next * example.is_img_match # is_img_match = 1 for mismatch images
    # if args.num_contrast_classes == 2 and args.texta_false_prob == 0.5 and is_next_type == 1:
    #     is_next_type = 2 # is_next_type 0: correct pair, 1: wrong text_b, 2: wrong text_a
    # if not args.mask_loss_for_unmatched and is_next_type == 2:
    #     t1_label = [-1]*len(tokens_a)
    # else:
    tokens_a, t1_label = random_word(tokens_a, tokenizer)
    if tokens_b:
        tokens_b, t2_label = random_word(tokens_b, tokenizer)
        # if not args.mask_loss_for_unmatched and is_next_type == 1:
        #     t2_label = [-1]*len(tokens_b)
        # else:
        #     tokens_b, t2_label = random_word(tokens_b, tokenizer)

    # else:
    #     theme_mask = [1] * len(theme_nodes) + [0] * (num_themes - len(theme_nodes))
    #     theme_nodes = theme_nodes + [0] * (num_themes - len(theme_nodes))

    if len(phrase_nodes) >= num_phrases+max_seq_length-2-len(tokens_a):
        phrase_nodes = phrase_nodes[:(num_phrases+max_seq_length-2-len(tokens_a))]
    phrase_mask = [1] * len(phrase_nodes)
    # else:
    #     phrase_mask = [1] * len(phrase_nodes) + [0] * (num_phrases - len(phrase_nodes))
    #     phrase_nodes = phrase_nodes + [0] * (num_phrases - len(phrase_nodes))

    # input id processing
    phrase_nodes, phrase_label = random_phrases(tokenizer, phrase_nodes, t1_label, phrase_mask_map)
    # theme_nodes, theme_label = random_theme(theme_nodes, tokenizer)

    # phrase_nodes = [p+tokenizer.vocab_size for p in phrase_nodes]
    # phrase_label = [p+tokenizer.vocab_size for p in phrase_label]
    phrase_label = [-1 for p in phrase_label]
    # theme_nodes = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_nodes]
    # theme_label = [t+tokenizer.vocab_size+phrase_vocab_size for t in theme_label]
    # theme_label = [-1 for p in theme_label]

    # concatenate lm labels and account for CLS, SEP, SEP
    # sequence_a (textual side)
    lm_label_ids_a = ([-1] + t1_label + phrase_label + [-1])
    lm_label_ids_b = ([-1] + t2_label + [-1])

    # if tokens_b:
    #     lm_label_ids_b = ([-1] + t1_label + phrase_label + [-1] + t2_label + theme_label + [-1])
    # else:
    #     lm_label_ids = ([-1] + t1_label + phrase_label + [-1])

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
    seq_tokens_a = []
    segment_ids_a = []
    seq_tokens_a.append("[CLS]")
    segment_ids_a.append(0)
    for token in tokens_a:
        seq_tokens_a.append(token)
        segment_ids_a.append(0)
    input_ids_a = tokenizer.convert_tokens_to_ids(seq_tokens_a)
    phrase_start_index = len(input_ids_a)
    phrase_end_index = phrase_start_index + len(phrase_nodes)

    for p in phrase_nodes:
        input_ids_a.append(p)
        segment_ids_a.append(0)

    input_ids_a.append(tokenizer.vocab["[SEP]"])
    segment_ids_a.append(0)

    seq_tokens_b = []
    segment_ids_b = []
    seq_tokens_b.append("[CLS]")
    segment_ids_b.append(1)
    if tokens_b:
        assert len(tokens_b) > 0
        segment_ids_b.extend([1]*len(tokens_b))
        seq_tokens_b.extend(tokens_b)
    
    input_ids_b = tokenizer.convert_tokens_to_ids(seq_tokens_b)

    input_ids_b.append(tokenizer.vocab["[SEP]"])
    segment_ids_b.append(1)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask_a = [1] * len(input_ids_a)
    input_mask_b = [1] * len(input_ids_b)

    # Zero-pad up to the sequence length.
    max_seq_length += num_phrases
    while len(input_ids_a) < max_seq_length:
        input_ids_a.append(0)
        input_mask_a.append(0)
        segment_ids_a.append(0)
        lm_label_ids_a.append(-1)

    while len(input_ids_b) < args.max_tag_length:
        input_ids_b.append(0)
        input_mask_b.append(0)
        segment_ids_b.append(1)
        lm_label_ids_b.append(-1)

    # s = "wrong length as maximal {}, input_ids {}, input_mask {}, segment{}, lm_label{}".format(max_seq_length, len(input_ids), len(input_mask), len(segment_ids), len(lm_label_ids))
    s = 'not valid sequence length, please check'
    assert len(input_ids_a) == max_seq_length, s
    assert len(input_mask_a) == max_seq_length, s
    assert len(segment_ids_a) == max_seq_length, s
    assert len(lm_label_ids_a) == max_seq_length, s

    # image features
    image_start_index = len(input_ids_a) # input_ids_a here for the concated sequence
    image_end_index = image_start_index + img_feat_len
    if args.max_img_seq_length > 0:
        if img_feat_len > args.max_img_seq_length:
            input_mask_b = input_mask_b + [1] * img_feat_len
        else:
            input_mask_b = input_mask_b + [1] * img_feat_len
            pad_img_feat_len = args.max_img_seq_length - img_feat_len
            input_mask_b = input_mask_b + ([0] * pad_img_feat_len)

    lm_label_ids_b = lm_label_ids_b + [-1] * args.max_img_seq_length

    if example.guid < 1:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens_a: %s" % " ".join([str(x) for x in seq_tokens_a]))
        logging.info("input_ids_a: %s" % " ".join([str(x) for x in input_ids_a]))
        logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_a]))
        logging.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_a]))
        logging.info("LM label seq A: %s " % lm_label_ids_a)
        logging.info("Is next sentence label: %s " % example.is_next)
        logging.info("tokens_b: %s" % " ".join([str(x) for x in seq_tokens_b]))
        logging.info("input_ids_b: %s" % " ".join([str(x) for x in input_ids_b]))
        logging.info("input_mask_b: %s" % " ".join([str(x) for x in input_mask_b]))
        logging.info("segment_ids_b: %s" % " ".join([str(x) for x in segment_ids_b]))
        logging.info("LM label seq B: %s " % lm_label_ids_b)
        # logging.info("Is next sentence label: %s " % example.is_next)

    features = InputFeatures(input_ids_a=input_ids_a,
                             input_mask_a=input_mask_a,
                             segment_ids_a=segment_ids_a,
                             lm_label_ids_a=lm_label_ids_a,
                             is_next=example.is_next,
                             qa_ans = qa_ans,
                             input_ids_b=input_ids_b,
                             input_mask_b=input_mask_b,
                             segment_ids_b=segment_ids_b,
                             lm_label_ids_b=lm_label_ids_b,
                             img_feat_len=img_feat_len,
                             is_img_match=example.is_img_match,
                             phrases_index = [phrase_start_index, phrase_end_index],
                             image_index = [image_start_index, image_end_index])
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

def text_concept_extract(text, concept_list):
    """TODO: how to extract concepts from the text, and the candidate list"""
    ## simple version 1, use high_frequence words + POS


class TextOnlyDataset(Dataset):
    def __init__(self, input_tsv, args, seq_len, tokenizer):
        if input_tsv.endswith('.tsv'):
            logging.info('Loading text only dataset under tsv format')
            self.is_tsv = True
            self.txt_tsv = TSVFile(input_tsv)
        else:
            logging.info('Loading text only dataset under huggingface datasets \
             format under {}'.format(input_tsv))
            self.is_tsv = False
            self.txt_tsv = datasets.load_from_disk(input_tsv, keep_in_memory=False)
            if hasattr(self.txt_tsv, 'keys'):
                # a dataset dict
                self.txt_tsv = self.txt_tsv['train']
        self.sample_count = 0
        self.args = args
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sample_counter = 0

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if self.is_tsv:
            row = self.txt_tsv.seek(item)
            txt_info = row[0].split('_')
            t1 = row[1]
        else:
            t1 = self.txt_tsv[item]['text']
            if item+1 < self.txt_tsv.num_rows:
                t1 += ' '+self.txt_tsv[item+1]['text']
        # print(item, row)

        t2 = ''
        is_next_label = -1
        is_img_match = -1

        # tokenize
        tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, self.seq_len-3)
        else:
            if len(tokens_a) > self.seq_len - 2:
                tokens_a = tokens_a[:(self.seq_len-2)]

        # transform sample to features
        tokens_a, t1_label = random_word(tokens_a, self.tokenizer)

        if tokens_b:
            if not self.args.mask_loss_for_unmatched and is_next_label == 1:
                t2_label = [-1]*len(tokens_b)
            else:
                tokens_b, t2_label = random_word(tokens_b, self.tokenizer)

        if tokens_b:
            lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
        else:
            lm_label_ids = [-1] + t1_label + [-1]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0]*len(tokens)

        if tokens_b:
            assert len(tokens_b) > 0
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1]*(len(tokens_b)+1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        # get image feature
        img_feat = torch.zeros(self.args.max_img_seq_length, self.args.img_feature_dim)
        img_feat_len = 0
        lm_label_ids = lm_label_ids + [-1] * self.args.max_img_seq_length
        input_mask += [0] * self.args.max_img_seq_length

        if self.args.visual_learning:
            target_img_feat = img_feat.clone()
            visual_labels = [-1]*self.args.max_img_seq_length
            mask_region_id = [0]*self.args.max_img_seq_length


        if cur_id <= 1:
            logging.info("*** Example ***")
            logging.info("guid: %s" % cur_id)
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("LM label: %s " % lm_label_ids)
            logging.info("Is next sentence label: %s " % is_next_label)



        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                item)
        else:
            return img_feat, (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match)
                ), item

    def __len__(self):
        if self.is_tsv:
            return len(self.txt_tsv)
        else:
            return self.txt_tsv.num_rows


class TextOnlyDataset2(Dataset):
    # text only dataset with full length as text
    def __init__(self, input_tsv, args, seq_len, tokenizer):
        print('text only dataset version V2!')
        if input_tsv.endswith('.tsv'):
            logging.info('Loading text only dataset under tsv format')
            self.is_tsv = True
            self.txt_tsv = TSVFile(input_tsv)
        else:
            logging.info('Loading text only dataset under huggingface datasets \
             format under {}'.format(input_tsv))
            self.is_tsv = False
            self.txt_tsv = datasets.load_from_disk(input_tsv)
            if hasattr(self.txt_tsv, 'keys'):
                # a dataset dict
                self.txt_tsv = self.txt_tsv['train']
        self.sample_count = 0
        self.args = args
        self.seq_len = seq_len + args.max_img_seq_length - 1
        self.img_seq_len = 1
        self.tokenizer = tokenizer
        self.sample_counter = 0

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if self.is_tsv:
            row = self.txt_tsv.seek(item)
            txt_info = row[0].split('_')
            t1 = row[1]
        else:
            t1 = self.txt_tsv[item]['text']
            tokens_a = self.tokenizer.tokenize(t1)
            p_id = 1
            while len(tokens_a)<self.seq_len-2 and item+p_id < self.txt_tsv.num_rows:
                # t1 += ' '+self.txt_tsv[item+1]['text']
                tokens_a += self.tokenizer.tokenize(self.txt_tsv[item+p_id]['text'])
                p_id += 1
                if p_id > 10:
                    break
                    print('looping for more than {} times now!'.format(p_id))
        # print(item, row)

        t2 = ''
        is_next_label = -1
        is_img_match = -1

        # tokenize
        # tokens_a = self.tokenizer.tokenize(t1)
        if self.args.use_b:
            tokens_b = self.tokenizer.tokenize(t2)
        else:
            tokens_b = None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, self.seq_len-3)
        else:
            if len(tokens_a) > self.seq_len - 2:
                tokens_a = tokens_a[:(self.seq_len-2)]

        # transform sample to features
        tokens_a, t1_label = random_word(tokens_a, self.tokenizer)

        if tokens_b:
            if not self.args.mask_loss_for_unmatched and is_next_label == 1:
                t2_label = [-1]*len(tokens_b)
            else:
                tokens_b, t2_label = random_word(tokens_b, self.tokenizer)

        if tokens_b:
            lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
        else:
            lm_label_ids = [-1] + t1_label + [-1]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0]*len(tokens)

        if tokens_b:
            assert len(tokens_b) > 0
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1]*(len(tokens_b)+1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        # get image feature
        if self.img_seq_len > 0:
            img_feat = torch.zeros(self.img_seq_len, self.args.img_feature_dim)
            img_feat_len = 0
            lm_label_ids = lm_label_ids + [-1] * self.img_seq_len
            input_mask += [0] * self.img_seq_len

            if self.args.visual_learning:
                target_img_feat = img_feat.clone()
                visual_labels = [-1]*self.img_seq_len
                mask_region_id = [0]*self.img_seq_len
        else:
            img_feat = None
            target_img_feat = None
            visual_labels = None
            mask_region_id = None

        if cur_id <= 1:
            logging.info("*** Example ***")
            logging.info("guid: %s" % cur_id)
            logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("LM label: %s " % lm_label_ids)
            logging.info("Is next sentence label: %s " % is_next_label)

        if self.args.deepspeed:
            return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                item)
        else:
            return img_feat, (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match)
                ), item

    def __len__(self):
        if self.is_tsv:
            return len(self.txt_tsv)
        else:
            return self.txt_tsv.num_rows
        


class ImgOnlyDataset(OscarTSVDataset_C):
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True,
                 **kwargs):
        super(ImgOnlyDataset, self).__init__(yaml_file=yaml_file, args=args,
        tokenizer=tokenizer, seq_len=seq_len, encoding=encoding, corpus_lines=corpus_lines,
        on_memory=on_memory, ds_names='oi_coco', kwargs=kwargs)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1

        sample = self.sample_to_doc[item]
        img_id = self.all_docs[sample['doc_id']][0].strip().split('|')[0]
        # t1 = self.all_docs[sample['doc_id']][sample['line']]
        t2 = self.all_docs[sample['doc_id']][sample['line']+1]
        # when using image-only dataset, no QA pairs are available
        assert('qa' not in self.all_docs[sample['doc_id']][0])

        img_feat = self.get_img_feature(img_id)
        img_feat_len = img_feat.shape[0]
        if img_feat.shape[0] >= self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length]
            img_feat_len = self.args.max_img_seq_length
            
        if self.args.visual_learning:
            target_img_feat = img_feat.clone()
            # print('-------')
            # print(self.all_docs[sample['doc_id']][0])
            img_feat, visual_labels, mask_region_id = random_visual(img_feat, t2, self.args.tag2id)
            # print('-------')

        if img_feat_len < self.args.max_img_seq_length:
            img_feat_len = img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.visual_learning:
                target_img_feat = torch.cat((target_img_feat, padding_matrix), 0)
                visual_labels += [-1]*(self.args.max_img_seq_length - img_feat_len)
                mask_region_id += [0]*(self.args.max_img_seq_length - img_feat_len)

        tokens_a = []
        tokens_b = self.tokenizer.tokenize(t2)
        _truncate_seq_pair(tokens_a, tokens_b, self.seq_len-3)

        # mask all text_a tokens
        tokens_b, t2_label = random_word(tokens_b, self.tokenizer)
        tokens = ["[CLS]"] + ['[SEP]'] + tokens_b + ['[SEP]']
        input_mask = [1]*len(tokens)
        segment_ids = [0]*2 + [1]*(len(tokens_b)+1)
        lm_label_ids = [-1]*2 + t2_label + [-1]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < self.seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        is_next_label = -1
        is_img_match = -1
        input_mask = input_mask + [1]*img_feat_len + [0]*(self.args.max_img_seq_length-img_feat_len)
        lm_label_ids = lm_label_ids + [-1] * self.args.max_img_seq_length
        
        if self.args.deepspeed:
            if self.args.visual_learning:
                return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                target_img_feat,
                torch.tensor(visual_labels, dtype=torch.long),
                torch.tensor(mask_region_id, dtype=torch.long),
                item)
            return (img_feat,
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match),
                item)
        else:
            return img_feat, (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor(lm_label_ids, dtype=torch.long),
                torch.tensor(is_next_label),
                torch.tensor(is_img_match)
                ), item




        
        
            
        

