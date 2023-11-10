from transformers import AutoTokenizer, AutoModel

# please clone the model bin files with utils code firstly to /aaa/bbb/ccc/ddd/chatglm3-6b
# env: py310, transformers==4.33.0

model_local_path = '/aaa/bbb/ccc/ddd/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_local_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_local_path, trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
