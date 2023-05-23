import json
import os

def load_ids(filename):
    ids = []
    for line in open(filename, 'r'):
        ids.append(line.strip())
    return set(ids)

def load_captions(filename, sep='\t'):
    caps = {}
    for line in open(filename, 'r'):
        info = line.strip().split(sep)
        img_key = info[0]
        caption = sep.join(info[1:])
        # try:
        #     img_key, caption = info
        # except:
        #     print(line, info)
        #     raise ValueError
        img_key = img_key.split('#')[0]
        if img_key in caps:
            caps[img_key].append(caption)
        else:
            caps[img_key] = [caption]
    return caps


def load_bosonseg(filename, sep=' '):
    caps = {}
    for line in open(filename, 'r'):
        info = line.strip().split(sep)
        img_key = info[0]
        words = info[1:]
        img_key = img_key.split('#')[0]
        caption = ''.join([w.split(':')[0] for w in words])
        if img_key in caps:
            caps[img_key].append(caption)
        else:
            caps[img_key] = [caption]
    return caps


train_ids = load_ids('coco-cn_train.txt')
val_ids = load_ids('coco-cn_val.txt')
test_ids = load_ids('coco-cn_test.txt')

human_written = load_captions('')