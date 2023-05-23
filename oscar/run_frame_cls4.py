# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
from __future__ import absolute_import, division, print_function
import argparse
from logging import raiseExceptions
import os
import base64
import os.path as op
import random, json
from typing import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from dataloader import KVReader
from tqdm import tqdm
import csv

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig 
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from sklearn.metrics import roc_curve, auc, precision_recall_curve


target_precision = 0.8

class ClassificationDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, data_in, is_train=True, i2v=None):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(ClassificationDataset, self).__init__()
        self.data_in = data_in
        self.i2v = i2v

        # get the image image_id to index map
        # imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        # self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
        
        # get the image features and labels
        self.img_feat_path = args.img_feat_path
        self.num_readers = args.num_readers
        
        if args.add_od_labels:
            self.od_tag_path = args.od_tag_path

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length + args.num_frames * args.tag_per_frame
        self.max_img_seq_len = args.max_img_seq_length + args.num_frames * args.obj_per_frame
        self.args = args


    def get_od_labels(self, img_key):
        if img_key.endswith('.jpg'):
            ii = img_key[:-4]
        else:
            ii = img_key
        vid = self.i2v[ii]
        img_key = vid + '.jpg'
        if self.args.add_od_labels:
            od_tags = self.od_reader.read_many([img_key])[0]
            od_tags = str(od_tags, encoding='utf-8')
            tmp = od_tags.split()
            if len(tmp) > self.args.max_img_seq_length:
                tmp = tmp[:self.args.max_img_seq_length]
                od_tags = ' '.join(tmp)
            return od_tags
        else:
            return None

    def get_full_tags(self, img_key):
        if img_key.endswith('.jpg'):
            ii = img_key[:-4]
        else:
            ii = img_key
        vid = self.i2v[ii]
        target_keys = [vid+'_'+str(i)+'.jpg' for i in range(self.args.num_frames)]
        return_values = self.od_reader.read_many(target_keys)
        res = []
        for v in return_values:
            tmp = str(v, encoding='utf-8').strip().split()
            if len(tmp) > self.args.tag_per_frame:
                tmp = tmp[:self.args.tag_per_frame]
            res.extend(tmp)
        return ' '.join(res)

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
                assert(num_extra_tokens >= 0)
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            for i, tk in enumerate(tokens):
                if tk not in self.tokenizer.special_tokens_map.values():
                    if random.random() <= self.args.random_mask:
                        tokens[i] = '[MASK]'
        if text_c:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)+text_c
            segment_ids += [sequence_c_segment_id]*len(text_c)
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                segment_ids = segment_ids[:self.max_seq_len]
            seq_len = len(input_ids)
            seq_padding_len = self.max_seq_len - seq_len
            segment_ids += [pad_token_segment_id] * seq_padding_len
            input_ids += self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token]*seq_padding_len)

        else:
            seq_padding_len = self.max_seq_len - seq_len
            tokens += [self.tokenizer.pad_token] * seq_padding_len
            segment_ids += [pad_token_segment_id] * seq_padding_len
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

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
        if self.is_train:
            for i in range(img_len):
                if random.random() <= self.args.random_mask:
                    img_feat[i, :] = 0

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

    def __getitem__(self, index):
        c_data = self.data_in[index]
        img_key = c_data[0]+'.jpg'
        img_feat = self.get_image(img_key)
        od_tags = self.get_od_labels(img_key)
        all_tags = self.get_full_tags(img_key)
        od_tags = od_tags + ' ' + all_tags
        frame_feat = self.get_frame_feat(img_key)
        img_feat = torch.cat([img_feat, frame_feat], dim=0)
        caption = c_data[2]
        label = c_data[1]
        if self.args.ex_con:
            text_c = c_data[-1]
        else:
            text_c = None
        example = self.tensorize_example(caption, img_feat, text_b=od_tags, text_c=text_c)
        return index, tuple(list(example) + [label])

    def get_image(self, image_id):
        if image_id.endswith('.jpg'):
            ii = image_id[:-4]
        else:
            ii = image_id
        vid = self.i2v[ii]
        image_id = vid+'.jpg'
        t_features = self.img_reader.read_many([image_id])[0]
        t_features = np.frombuffer(t_features, dtype=np.float32).reshape(-1, self.args.img_feature_dim)
        if not t_features.flags['WRITEABLE']:
            new_features = np.copy(t_features)
        else:
            new_features = t_features
        if new_features.shape[0] > self.args.max_img_seq_length:
            new_features = new_features[:self.args.max_img_seq_length, :]
        new_features = torch.tensor(new_features, dtype=torch.float)
        return new_features

    def get_frame_feat(self, image_id):
        if image_id.endswith('.jpg'):
            ii = image_id[:-4]
        else:
            ii = image_id
        vid = self.i2v[ii]
        target_keys = [vid+'_'+str(i)+'.jpg' for i in range(self.args.num_frames)]
        return_values = self.img_reader.read_many(target_keys)
        res = []
        for k in return_values:
            t_features = np.frombuffer(k, dtype=np.float32).reshape(-1, self.args.img_feature_dim)
            if not t_features.flags['WRITEABLE']:
                new_features = np.copy(t_features)
            else:
                new_features = t_features
            if new_features.shape[0] > self.args.obj_per_frame:
                new_features = new_features[:self.args.obj_per_frame, :]
            new_features = torch.tensor(new_features, dtype=torch.float)
            res.append(new_features)
        return torch.cat(res, dim=0)
            

    def get_label(self, index):
        return self.data_in[index][1]

    def get_item_id(self, index):
        return self.data_in[index][0]

    def __len__(self):
        return len(self.data_in)

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.img_reader = KVReader(dataset.img_feat_path, dataset.num_readers)
    dataset.od_reader = KVReader(dataset.od_tag_path, dataset.num_readers)

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


def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

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

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4]
            }
            if args.loss_weight:
                inputs['loss_weights'] = get_loss_weight(args).to(args.device)
            inputs = prepare_inputs(inputs, args)
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_score = compute_score_with_logits(logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.train_batch_size)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        batch_acc, global_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        test_result = test(args, model, val_dataset)
                        test_result = {val_dataset.get_item_id(k):(v, val_dataset.get_label(k)) for k,v in test_result.items()}
                        torch.save(test_result, args.output_dir + '/predictions-{}-{}.pth'.format(epoch, global_step))
                        eval_result = evaluate(test_result)
                        if eval_result['AUC'] > best_score:
                            best_score = eval_result['AUC']
                        epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                     'accuracy': eval_result['accuracy'], 'precision': eval_result['precision'], 
                                     'recall': eval_result['recall'], 'AUC': eval_result['AUC'], 
                                     'R(P>=0.9)': eval_result['R(P>=0.9)'], 'threshold': eval_result['threshold'],'best_auc':best_score}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
    return global_step, global_loss / global_step

def prepare_inputs(inputs, args):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if inputs[k].dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                inputs[k]=v.to(dtype=args.dtype)
    return inputs

def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
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
                result = probs[:, 1] # the confidence to be a positive pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]
            results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
    return results


def evaluate(test_results, th = None):
    total = 0
    res = {(0, 0):0, (0, 1):0, (1, 0):0, (1, 1):0}
    all_labels = []
    all_preds = []
    if th is None:
        th = 0.5
    for k,v in test_results.items():
        total += 1
        label = v[1]
        logit = v[0]
        all_labels.append(label)
        all_preds.append(logit)
        pred = int(logit > th)
        res[(label, pred)] += 1
    acc = (res[(0, 0)]+res[(1, 1)])/total
    prec = res[(1, 1)]/(res[(0, 1)]+res[(1, 1)])
    recall = res[(1, 1)]/(res[(1, 0)]+res[(1, 1)])
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds, pos_label=1)
    lr_precision, lr_recall, threshs = precision_recall_curve(all_labels, all_preds)
    r_pthre = 0
    my_th = 0
    for i in range(len(lr_precision)):
        if lr_precision[i] >= target_precision:
            r_pthre = lr_recall[i]
            print('Recall (Precision>={}):'.format(target_precision), lr_recall[i])
            my_th = threshs[i]
            break
    auc_score = auc(fpr, tpr)
    logger.info("Binary Classification: Accuracy: {:.4f}, Precision {:.4f} , Recall {:.4f} , AUC {:.4f}, R(P>={}) {:.4f}, threshold {:.4f}".format(
                acc, prec, recall, auc_score, target_precision, r_pthre, my_th))
    eval_result = {"accuracy": acc, "recall": recall, "precision": prec, "AUC": auc_score, "R(P>=0.9)": r_pthre, "threshold": my_th}
    
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
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args

def get_loss_weight(args):
    raw_weights = torch.FloatTensor([1, 2])
    all_weights = torch.cat([raw_weights]*args.n_gpu, dim=0)
    return all_weights

def filter_ids(filter_file):
    filter_dict = torch.load(filter_file)
    good_ids = set()
    for k,v in filter_dict.items():
        if v[1] == 1 and v[0] > 0.7:
            good_ids.add(k)
        elif v[1] == 0 and v[0] < 0.3:
            good_ids.add(k)
    del(filter_dict)
    return good_ids



def label_process(lab, mod='origin'):
    """
    a function to apply pre-processing on labels
    lab in {1, 2, 3}
    """
    lab = int(lab)
    if mod=='origin':
        # keep the origin classes
        return lab - 1
    elif mod == 'reduce':
        # reduce to binary
        if lab in [1, 2]:
            return 1
        else:
            return 0
    else:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_path", default=None, type=str, help='the path to input image features')
    parser.add_argument("--od_tag_path", default=None, type=str, help='the path to input object detection tags')
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
    parser.add_argument('--num_extra_concept', type=int, default=5, help="Number of extra concapts added")
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help="Which GPUs to use")
    parser.add_argument('--half_evaluation', action='store_true', help='Whther to use half precision for evaluation')
    parser.add_argument('--num_readers', type=int, default=1, help='number of readers for kvreader')
    parser.add_argument('--data_frame', type=str, default=None, help='input data frame')
    parser.add_argument('--val_split', type=str, default=None, help='validation split indices file')
    parser.add_argument('--val_split2', type=str, default=None, help='validation split-2 indices file')
    parser.add_argument('--label_mod', type=str, default='origin', help='which method used to transform labels')
    parser.add_argument('--random_mask', default=0, type=float, help='probability to mask some words or regions during training')
    parser.add_argument('--loss_weight', action='store_true', help='whether to use weighted corss entropy to handel inbalance')
    parser.add_argument('--ex_con', action='store_true', help='wether to use extra concept')
    parser.add_argument('--item2tag', type=str, default=None, help='item 2 tags json')
    parser.add_argument('--tag2id', type=str, default=None, help='tag 2 id json')
    parser.add_argument('--dev_prop', type=float, default=0.1, help='development set proportion')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold during evaluation')
    parser.add_argument('--target_precision', type=float, default=0.8, help='the target precision to control')
    parser.add_argument('--obj_per_frame', type=int, default=15, help='number of objects used per frame')
    parser.add_argument('--tag_per_frame', type=int, default=0, help='number of object tags per frame')
    parser.add_argument('--item2vid', type=str, default=None, help='the item id 2 vid mapping json')
    parser.add_argument('--num_frames', type=str, default=5, help='number of frames used per video')
    parser.add_argument('--belief_data', type=str, default=None, help='high belief data (labeled)')
    parser.add_argument('--filter_file', type=str, default=None, help='early predictions file')
    parser.add_argument('--refined', action='store_true', help='whether the model is re-trained')
    parser.add_argument('--eval_split', type=str, default=None, help='which split to evaluate on')
    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    global target_precision
    target_precision = args.target_precision

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification

    # ensure label num
    if args.ex_con:
        with open(args.tag2id, 'r') as rf:
            tag2id = json.load(rf)
        with open(args.item2tag, 'r') as rf:
            item2tag = json.load(rf)
    if args.label_mod == 'reduce':
        args.num_labels = 2
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
        if args.ex_con:
            if not args.refined:
                config.vocab_size += len(tag2id)
        model = model_class.from_pretrained(args.model_name_or_path, 
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        model.reinit_cls_head()
        if args.half_evaluation:
            model = model.half()
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        if args.half_evaluation:
            model = model.half()
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # process validation split
    if args.val_split.split('/')[-1] == 'NULL':
        args.val_split = None
    test_ids = set()
    if args.val_split:
        test_split = open(args.val_split, 'r')
        tsp_csv = csv.reader(test_split)
        next(tsp_csv)
        for line in tsp_csv:
            test_ids.add(line[1])
    if args.val_split2:
        test_split = open(args.val_split2, 'r')
        tsp_csv = csv.reader(test_split)
        next(tsp_csv)
        for line in tsp_csv:
            test_ids.add(line[0])

    # load item id 2 vid
    with open(args.item2vid, 'r') as rf:
        my_item2vid = json.load(rf)
        add_i2v = json.load(open(os.path.split(args.item2tag)[0]+'/add_i2v_V3.json', 'r'))
        for k,v in my_item2vid.items():
            if k in add_i2v:
                continue
            my_item2vid[k] = v[2:]

    if args.belief_data is not None:
        with open(args.belief_data, 'r') as rf:
            hb_data = set([i.strip() for i in rf.readlines()])
    else:
        hb_data = None

    if args.filter_file is not None:
        # filter_file is used during training for training set filtering
        good_ids = filter_ids(args.filter_file)
    else:
        good_ids = None
    
    train_data = []
    dev_data = []
    val_data = []
    data_file = open(args.data_frame, 'r')
    data_csv = csv.reader(data_file)
    next(data_csv)
    for line in data_csv:
        if line[0] not in test_ids:
            if hb_data is not None:
                if line[0] in hb_data:
                    continue
            if good_ids is not None:
                if line[0] not in good_ids:
                    continue
        if line[0] not in my_item2vid:
            continue
        new_line = [info for info in line]
        new_line[1] = int(new_line[1])
        if args.ex_con:
            tmp = [tag2id[i] for i in item2tag[line[0]].split(';') if i in tag2id]
            # print(tmp)
            new_line.append(tmp)
        if line[0] in test_ids:
            val_data.append(new_line)
        else:
            if random.random()<args.dev_prop:
                dev_data.append(new_line)
            else:
                train_data.append(new_line)

    train_label_prop = Counter([line[1] for line in train_data])
    val_label_prop = Counter([line[1] for line in val_data])
    print('training label proportion:', train_label_prop.most_common())
    print('validation label proportion:', val_label_prop.most_common())
    if args.do_train:
        train_dataset = ClassificationDataset(tokenizer, args, train_data, is_train=True, i2v=my_item2vid)
        if args.evaluate_during_training:
            val_dataset = ClassificationDataset(tokenizer, args, val_data, is_train=False, i2v=my_item2vid)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        if args.eval_split == 'test':
            vd = val_data
        else:
            vd = train_data
        test_dataset = ClassificationDataset(tokenizer, args, vd, is_train=False, i2v=my_item2vid)
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        if args.half_evaluation:
            model = model.half()
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        test_result = test(args, model, test_dataset)
        test_result = {test_dataset.get_item_id(k):(v, test_dataset.get_label(k)) for k,v in test_result.items()}
        torch.save(test_result, args.output_dir + '/predictions-test.pth')
    if args.do_eval:
        # do evaluation
        eval_result = evaluate(test_result, th=args.threshold)
        result_file = args.output_dir + 'test.eval.json'
        with open(result_file, 'w') as f:
            json.dump(eval_result, f)
        logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
