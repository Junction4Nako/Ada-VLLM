import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random
from oscar.utils.misc import mkdir, set_seed
from oscar.utils.logger import setup_logger
from torch.optim import SGD
import os.path as op
from tqdm import tqdm
import json, os
from transformers_past.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


def close_form_solution(src, tar):
    tmp = torch.matmul(src.t(), src)
    inv_tmp = torch.inverse(tmp)
    return torch.matmul(torch.matmul(inv_tmp, src.t()), tar)

class WordTranslationDataset(Dataset):
    def __init__(self, source_embs, target_embs):
        assert len(source_embs) == len(target_embs), 'source embeddings should be aligned with target embeddings'
        self.source_embs = source_embs
        self.target_embs = target_embs

    def __getitem__(self, index):
        return self.source_embs[index], self.target_embs[index]

    def __len__(self):
        return len(self.source_embs)

class WordMapping(nn.Module):
    def __init__(self, model_type, input_dim, output_dim, dropout=0):
        super(WordMapping, self).__init__()
        self.model_type = model_type
        assert self.model_type in ['mlp', 'linear'], 'only support linear or mlp mapping'
        if self.model_type == 'mlp':
            self.map = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(output_dim, output_dim)
            )
        else:
            self.map = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(input_dim, output_dim)
            )
    
    def forward(self, input_emb, target_emb=None):
        pred_embs = self.map(input_emb)
        if target_emb is None:
            return pred_embs
        else:
            diff_norm = torch.norm(pred_embs-target_emb, p=2, dim=-1)**2
            return pred_embs, torch.mean(diff_norm)


def emb_encode(word, tokenizer, embedding):
    unk_token = tokenizer.special_tokens_map['unk_token']
    sub_tokens = tokenizer.tokenize(word)
    sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)
    if unk_token in sub_tokens:
        # ignore UNK token
        return None
    if len(sub_tokens) > 1:
        # ignore multiple sub-words
        return None
    return embedding[sub_tokens[0],:]

def save_checkpoint(model, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            # model_to_save.save_pretrained(checkpoint_dir)
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'pytorch_model.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            # tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, model, train_dataset, val_dataset):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    if args.scheduler == 'constant':
        scheduler = WarmupConstantSchedule(
            optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == 'linear':
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise NotImplementedError
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

    global_step, global_loss = 0, 0.0
    model.zero_grad()
    best_loss = 9999
    log_json = []
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_emb':   batch[0],
                'target_emb':  batch[1]
            }
            pred, loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_loss = loss.item()
            global_loss += batch_loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, \
                        global_step,optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step))
                    
                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        val_loss = val(args, model, val_dataset)
                        if val_loss < best_loss:
                            best_loss = val_loss
                        logger.info('Word Translation in epoch {}, val loss: {}'.format(epoch, val_loss))
                        epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                     'best_loss':best_loss}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 

def val(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    total_loss = 0.0
    total_size = 0
    for batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_emb':   batch[0],
                'target_emb':  batch[1]
            }
            pred, loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()
            batch_size = batch[0].shape[0]
            total_size += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_size



def main():
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_emb', type=str, default=None, help='source embedding weight')
    parser.add_argument('--source_tokenizer', type=str, default=None, help='source tokenizer')
    parser.add_argument('--target_emb', type=str, default=None, help='target embedding weight')
    parser.add_argument('--target_tokenizer', type=str, default=None, help='target tokenizer')
    parser.add_argument('--dict_file', type=str, default=None, help='the dictionary file')
    parser.add_argument('--mapping_mod', type=str, default='same_word', help='the word translation mapping mod')
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument('--loss_type', type=str, default='mse', help='the loss type mse or InfoNCE')
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument('--mapping_type', default='linear', type=str, help='the word mapping module type')
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--word_num_limit', type=int, default=-1, help='the number limit of tokens')
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument('--seed', type=int, default=88, help='random seed')
    parser.add_argument('--val_prop', type=float, default=0.1, help='the proportion of the validtion set')
    parser.add_argument('--no_cuda', action='store_true', help='not using cuda')
    parser.add_argument('--closed_form', action='store_true', help='using the closed form method')

    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    
    assert args.mapping_mod in ['same_word', 'cross_lingual'], 'mapping mod must be based on same words or a cross-lingual dictionary'

    # loading the tokenizer
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)

    # loadint the embedding
    source_emb = torch.load(args.source_emb, map_location='cpu')
    if isinstance(source_emb, dict):
        valid_key = None
        for k in source_emb.keys():
            if 'word_embeddings.weight' in k:
                valid_key = k
        source_emb = source_emb[valid_key]
    source_emb = source_emb.to(torch.float32)
    target_emb = torch.load(args.target_emb, map_location='cpu')
    if isinstance(target_emb, dict):
        valid_key = None
        for k in target_emb.keys():
            if 'word_embeddings.weight' in k:
                valid_key = k
        target_emb = target_emb[valid_key]
    target_emb = target_emb.to(torch.float32)

    # loading the dictionary
    if args.mapping_mod == 'same_word':
        train_emb_pairs = []
        val_emb_pairs = []
        c = 0
        # the dictionary file should contain multiple lines, each line contains one word
        with open(args.dict_file, 'r') as rf:
            for line in tqdm(rf):
                word = line.strip()
                s_emb = emb_encode(word, source_tokenizer, source_emb)
                t_emb = emb_encode(word, target_tokenizer, target_emb)
                if s_emb is None or t_emb is None:
                    continue
                if args.word_num_limit > 0 and c >= args.word_num_limit:
                    break
                if random.random() < 1-args.val_prop:
                    train_emb_pairs.append([s_emb, t_emb])
                else:
                    val_emb_pairs.append([s_emb, t_emb])
                c += 1
    
    elif args.mapping_mod == 'cross_lingual':
        train_emb_pairs = []
        val_emb_pairs = []
        c = 0
        # the dictionary file should contain multiple lines, each line contains two words seperate by comma
        with open(args.dict_file, 'r') as rf:
            for line in rf:
                word = [w.strip() for w in line.strip().split(',')]
                s_emb = emb_encode(word[0], source_tokenizer, source_emb)
                t_emb = emb_encode(word[1], target_tokenizer, target_emb)
                if s_emb is None or t_emb is None:
                    continue
                if args.word_num_limit > 0 and c >= args.word_num_limit:
                    break
                if random.random() < 1-args.val_prop:
                    train_emb_pairs.append([s_emb, t_emb])
                else:
                    val_emb_pairs.append([s_emb, t_emb])
                c += 1

    else:
        raise NotImplementedError

    special_tokens = ['unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']
    for s_tok in special_tokens:
        s_emb = emb_encode(source_tokenizer.special_tokens_map[s_tok], source_tokenizer, source_emb)
        t_emb = emb_encode(target_tokenizer.special_tokens_map[s_tok], target_tokenizer, target_emb)
        train_emb_pairs.append([s_emb, t_emb])

    logger.info('founded {} pairs'.format(len(train_emb_pairs)+len(val_emb_pairs)))

    if args.closed_form:
        src_full_embs = []
        tar_full_embs = []
        for pair in train_emb_pairs + val_emb_pairs:
            src_full_embs.append(pair[0])
            tar_full_embs.append(pair[1])
        src_full_embs = torch.stack(src_full_embs)
        tar_full_embs = torch.stack(tar_full_embs)
        map_W = close_form_solution(src_full_embs, tar_full_embs)
        torch.save(map_W, os.path.join(args.output_dir, 'word_mapping.pth'))
        return None
    
    # dataset construct
    train_source, train_target = zip(*train_emb_pairs)
    train_dataset = WordTranslationDataset(train_source, train_target)

    val_source, val_target = zip(*val_emb_pairs)
    val_dataset = WordTranslationDataset(val_source, val_target)

    # initialize model
    logger.info('source embedding dim {}, target embedding dim {}'.format(source_emb.shape[1], target_emb.shape[1]))
    model = WordMapping(args.mapping_type, source_emb.shape[1], target_emb.shape[1])
    model.to(args.device)

    train(args=args, model=model, train_dataset=train_dataset, val_dataset=val_dataset)


if __name__=='__main__':
    main()