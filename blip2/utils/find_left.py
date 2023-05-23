import os
import shutil
import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=None)
    parser.add_argument('--index_file', type=str, default=None)
    parser.add_argument('--target_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.index_file, 'r') as rf:
        valid = [line.strip() for line in rf]
    valid = set(valid)
    for fn in tqdm.tqdm(os.listdir(args.source_dir)):
        if fn in valid:
            continue
        else:
            shutil.copyfile(os.path.join(args.source_dir, fn), os.path.join(args.target_dir, fn))

if __name__=='__main__':
    main()