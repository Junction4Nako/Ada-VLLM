import argparse
import torch
from lavis.models.blip2_models.blip2_llama import Blip2Llama
from PIL import Image
import yaml
from transformers import AutoTokenizer, AutoConfig
from torchvision import transforms
from oscar.utils.randaugment import RandomAugment

def create_transform(config, name='pretrain'):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ]) 
    if name == 'pretrain':
        return pretrain_transform
    elif name == 'train':
        return train_transform
    else:
        return test_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--image', type=str, default='None')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # loading and initializing
    img = Image.open(args.image)
    img = img.convert('RGB')
    config = yaml.load(open(args.model_config, 'r'), Loader=yaml.Loader)
    if 'local_ckpt' in config:
        local_ckpt_cfg = config['local_ckpt']
    else:
        local_ckpt_cfg = None
    # adpat arguments from config to arguments
    args.image_size = config['model']['image_size']
    args.model_config = config
    adaptive_llm = config['adaptive_llm']
    tokenizer = AutoTokenizer.from_pretrained(adaptive_llm)
    ada_config = AutoConfig.from_pretrained(adaptive_llm)

    if tokenizer.pad_token is None:
        # tokenizer.add
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        ada_config.vocab_size += 1

    model = Blip2Llama(vit_model=config['model']['vit_model'],
                        img_size=config['model']['image_size'],
                        drop_path_rate=config['model']['drop_path_rate'],
                        use_grad_checkpoint=config['model']['use_grad_checkpoint'],
                        vit_precision=config['model']['vit_precision'],
                        freeze_vit=config['model']['freeze_vit'],
                        num_query_token=config['model']['num_query_token'],
                        max_txt_len=config['model']['max_txt_len'],
                        ada_config=ada_config,
                        ada_tokenizer=tokenizer,
                        llm_model=config['model']['llm'],
                        local_ckpt_cfg=local_ckpt_cfg,
                        qformer_text_input=False)
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    trans_cfg = {'image_res': config['image_size']}
    image_transform = create_transform(trans_cfg, 'test')
    img = image_transform(img)


    with torch.no_grad():
        model.eval()
        device = torch.device(args.device)
        img = img.unsqueeze(0).to(device)
        samples = {'image': img, 'prompt': args.text}
        res = model.generate(samples)
        print(res)

if __name__=='__main__':
    main()
