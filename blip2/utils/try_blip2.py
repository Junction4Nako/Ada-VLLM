from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

config = BertConfig.from_pretrained('bert-base-uncased')
config.encoder_width = 768
config.add_cross_attention = True
config.cross_attention_freq = 2
config.query_length = 20

config.adaptive = True
config.target_vocab_size = 32000
config.target_hidden_size = 4096
config.lmhead_bias = False

# perform embedding mapping using anchor tokens
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, BertModel
import torch

llama_tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model, bert_msg = BertModel.from_pretrained('bert-base-uncased', output_loading_info=True)
bert_embedding = bert_model.embeddings.word_embeddings.weight
llama_embeddings = torch.load('/remote-home/zjli/LLM/checkpoints/llama/llama_7b_word_embeddings.pt')
pairs = []
for k,v in bert_tokenizer.vocab.items():
    if '##' in k:
        continue
    sub_tokens = llama_tokenizer.tokenize(k)
    sub_tokens = llama_tokenizer.convert_tokens_to_ids(sub_tokens)
    if len(sub_tokens) == 1:
        pairs.append((v, sub_tokens[0]))

bert_tmp = []
llama_tmp = []
for p in pairs:
    bert_tmp.append(bert_embedding[p[0]])
    llama_tmp.append(llama_embeddings[p[1]])

bert_tmp = torch.stack(bert_tmp, dim=0)
llama_tmp = torch.stack(llama_tmp, dim=0)

def close_form_solution(src, tar):
    tmp = torch.matmul(src.t(), src)
    inv_tmp = torch.inverse(tmp)
    return torch.matmul(torch.matmul(inv_tmp, src.t()), tar)

bert2llama = close_form_solution(bert_tmp, llama_tmp)
llama2bert = close_form_solution(llama_tmp, bert_tmp)

import torch
import torch.nn as nn
from torch.optim import SGD

class test(nn.Module):
    def __init__(self, input_dim=4, out=10):
        super(test, self).__init__()
        self.mid_map = nn.Linear(input_dim, input_dim)
        self.map = nn.Linear(input_dim, out)

    def forward(self, input_x):
        out_x = self.map(self.mid_map(input_x))
        return out_x[:, :-1].contiguous()
    
input_dim = 4
out = 5
model = test(input_dim=input_dim, out=out)
optimizer = SGD(model.parameters(), lr=0.1)
model.train()
input_x = torch.randn(4, input_dim)
label = torch.randint(0, out, (4,))
ce = nn.CrossEntropyLoss()

out_x = model(input_x)
loss = ce(out_x, label)
loss.backward()