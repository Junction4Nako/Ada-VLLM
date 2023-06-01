from transformers import AutoTokenizer
import json
import argparse
import tqdm
import os
from multiprocessing import Pool

tokenizer=None

def token_count(fn_name):
    n_tokens = 0
    n_lines = 0
    with open(fn_name, 'r') as rf:
        for line in rf:
            n_lines += 1
            data = json.loads(line.strip())
            n_tokens += len(tokenizer.tokenize(data['text']))
    print('founded {} lines with {} tokens in {}'.format(n_lines, n_tokens, fn_name))
    return n_tokens

def line_count(fn_name):
    n_tokens = 0
    n_lines = 0
    with open(fn_name, 'r') as rf:
        for line in rf:
            n_lines += 1
    print('founded {} lines with in {}'.format(n_lines, fn_name))
    return n_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default=None, help='the tokenizer to use')
    parser.add_argument('--dir_name', type=str, default=None, help='the directory name')
    parser.add_argument('--num_threads', type=int, default=20, help='the number of threads')
    parser.add_argument('--count', type=str, default='token', help='count line or tokens')
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    if args.count == 'line':
        count_fn = line_count
    elif args.count == 'token':
        count_fn = token_count
    else:
        raise ValueError

    all_fns = [os.path.join(args.dir_name, fn) for fn in os.listdir(args.dir_name) if fn.endswith('json') or fn.endswith('jsonl')]
    
    final_res = 0
    with Pool(args.num_threads) as p:
        for res in tqdm.tqdm(p.imap(count_fn, all_fns)):
            final_res += res

    print('found {} {}s in {}'.format(final_res, args.count, args.dir_name))

if __name__=='__main__':
    main()
