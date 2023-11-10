#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # run at gpu num 2; change to "0,1,2" if you have 3 gpu

from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import torch


base_model_path = "THUDM/chatglm2-6b"
cache_path = '/data1/yinbin/projects/huggingface_cache' # huggingface will download model to this cache folder if not find at base_model_path
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, cache_dir=cache_path)
config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, pre_seq_len=128, cache_dir=cache_path)
model = AutoModel.from_pretrained(base_model_path, config=config, trust_remote_code=True, cache_dir=cache_path)
check_point_path = os.path.join("/aaa/bbb/ccc/ddd/ChatGLM2-6B-main/ptuning/output/my-1k-chatglm2-6b-pt-128-1e-2/checkpoint-200/", "pytorch_model.bin")
prefix_state_dict = torch.load(check_point_path)
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)


#print(f"Quantized to 4 bit")
#model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()



ques = r"""who are you?"""

response, history = model.chat(tokenizer, ques, temperature=0.2, history=[])
print(response)
