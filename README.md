# llm_learning
all about large language related


# 1. model basic and config
- [clone llm bin files and utils code](misc/clone_model.md)
- [set run at gpu num](https://github.com/ybdesire/llm_learning/blob/main/ptuning_chatglm2/load_ft_model_for_inference.py#L5)


# 2. chatglm2 inference
- [load base model for inference](ptuning_chatglm2/load_raw_model_inference.py)


# 3. chatglm2 ptuning

- [ptuning steps](ptuning_chatglm2/ptuning_steps_chatglm2.md)
- [1GPU.fp16](ptuning_chatglm2/train_chat.1GPU.fp16.sh)
- [1GPU.int4](ptuning_chatglm2/train_chat.1GPU.int4.sh)
- [3GPU.fp16](ptuning_chatglm2/train_chat.3GPU.fp16.sh)

# 4. chatglm2 load ptuning model for inference
- [load ptuning model checkpoint for inference](ptuning_chatglm2/load_ft_model_for_inference.py)


# 5. LLaMA-Factory
