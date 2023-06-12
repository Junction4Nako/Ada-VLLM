import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import os, json
import tqdm
import logging
import re
from PIL import Image

class Llava_Instruct(Dataset):
    def __init__(self, data_path, image_dir, image_transform, header=None, line_sep=None):
        self.image_dir = image_dir
        self.data = []
        self.image_transform = image_transform
        self.line_sep = line_sep if line_sep is not None else ' '
        logging.info('using "{}" as the seperator between lines'.format(self.line_sep))
        if header is None:
            self.user_header = '### User: '
            self.assistant_header = '### Assistant: '
        else:
            self.user_header = header['user']
            self.res_header = header['assist']

        if os.path.isdir(data_path):
            # local path
            conversastion = json.load(open(os.path.join(data_path, 'conversation_58k.json')))
            complex_reason = json.load(open(os.path.join(data_path, 'complex_reasoning_77k.json')))
            detail = json.load(open(os.path.join(data_path, 'detail_23k.json')))
            logging.info('preprocessing the conversation data')
            self.preprocess_data(conversastion)
            logging.info('preprocessing the complex reasoning data')
            self.preprocess_data(complex_reason)
            logging.info('preprocessing the detailed caption data')
            self.preprocess_data(detail)
        else:
            tmp_data = load_dataset(data_path)
            logging.info('preprocessing the full data from huggingface')
            self.preprocess_data(tmp_data['train'])
        logging.info('we have {} samples in Llava-Instruct to train!'.format(len(self.data)))

    def preprocess_data(self, data_list):
        for line in tqdm.tqdm(data_list):
            image_fn = self.get_image_name(line['image'])
            convs = line['conversations']
            assert len(convs) % 2 == 0, 'the instruct format must be 2 turns'
            history_data = ''
            for i in range(len(convs) // 2):
                row = {'image': image_fn}
                current_query = convs[i*2]
                current_response = convs[i*2 + 1]
                assert current_query['from'] == 'human', 'the input query must come from human!'
                assert current_response['from'] == 'gpt', 'the output response must come from GPT!'
                # remove the image placehold in query
                query_str = re.sub('<image>', '', current_query['value'])
                query_str = re.sub('\n', '', query_str) + self.line_sep
                
                # make the query and output
                query_str = self.user_header + query_str + self.assistant_header
                response_str = current_response['value']

                # concate history with the current
                if history_data == '':
                    text_input = query_str
                    history_data = query_str + response_str + self.line_sep
                else:
                    text_input = history_data + query_str
                    history_data = history_data + query_str + response_str + self.line_sep
                row['text_input'] = text_input
                row['text_output'] = response_str
                self.data.append(row)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        row['image'] = self.get_image_feature(row['image'])
        return row

    def get_image_feature(self, image_fn):
        img = Image.open(image_fn)# .convert('RGB')
        try:
            img = self.image_transform(img)
        except:
            img = self.image_transform(img.convert('RGB'))
        return img

    def get_image_name(self, image_id):
        # transform the image id into COCO format
        return os.path.join(self.image_dir, 'COCO_train2014_{}'.format(image_id))


if __name__ == '__main__':
    ds = Llava_Instrcuct('liuhaotian/LLaVA-Instruct-150K')