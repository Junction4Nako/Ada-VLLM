from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from albef.modeling.model_pretrain import ALBEF_Stage1
import yaml
from xvlm.modeling.xroberta import RobertaForMaskedLM

device = torch.device('cuda:7')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
xlm_roberta_base = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
test_texts = torch.load('/remote-home/zjli/tmp_debug/cc_100_test.pt')
text_input = tokenizer(test_texts['text'], padding='longest', truncation=True, max_length=64, return_tensors="pt").to(device)

config = yaml.load(open('albef/configs/pretrain_base_xlm-r_unistage2_freeze_vis.yaml', 'r'), Loader=yaml.Loader)
model = ALBEF_Stage1(config=config, text_encoder=config['text_encoder'], tokenizer=tokenizer, init_deit=True)

ckpt_file = '/remote-home/zjli/tmp_data/xlm-roberta-base/albef_xlmr_roberta_base_ver2.pt'
checkpoint = torch.load(ckpt_file, map_location='cpu')
target_keys = set(model.state_dict().keys())
# tmp_keys = ['text_encoder.bert.embeddings.word_embeddings.weight', 'text_encoder.cls.predictions.decoder.bias', 'text_encoder.cls.predictions.decoder.weight', 'text_encoder.cls.predictions.bias']
tmp_keys = []
try:
    state_dict = {k:v for k,v in checkpoint.items() if k in target_keys and k not in tmp_keys}
except:
    state_dict = {k:v for k,v in checkpoint['model'].items() if k in target_keys and k not in tmp_keys}

msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model.to(device)
model.eval()
xlm_roberta_base.to(device)
xlm_roberta_base.eval()

with torch.no_grad():
    my_model_output = model.forward_mono_txt_debug(text_input)
    new_input_ids, labels = my_model_output[1]
    print('my_res')
    print(my_model_output[0][1].loss)

    lm_mask = labels >= 0

    old_res = xlm_roberta_base(new_input_ids, attention_mask=text_input.attention_mask, labels=labels, output_hidden_states=True)
    print('old result')
    print(old_res.loss)
    # num_tokens = old_res.logits.shape[-1]
    # print(torch.masked_select(old_res.logits, lm_mask.unsqueeze(-1)).reshape(-1, num_tokens))
    # num_dims = old_res.hidden_states.shape[-1]
    # print(torch.masked_select(old_res.hidden_states, lm_mask.unsqueeze(-1)).reshape(-1, num_dims))

    torch.save({'input_ids': new_input_ids, 'labels':labels, 'my_res': my_model_output[0], 'old_res': old_res}, '/remote-home/zjli/tmp_debug/debug_res.pth')

