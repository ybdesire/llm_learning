# llm_learning
all about LLM (large language model)


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
- [steps to use LLaMA-Factory fine tuning framework](llama_factory/llama_factory_usage_steps.md)
- [fine tuning chatglm2.lora.1gpu.fp16](llama_factory/train.chatglm2.lora.1gpu.fp16.sh)
- [load model cli demo chatglm2.lora.1gpu](llama_factory/cli.chatglm2.lora.1gpu.sh)
- [support multiple dataset](https://github.com/hiyouga/LLaMA-Factory/issues/1297)
- [mulitple GPU on one node by accelerate](llama_factory/accelerate_4gpu_sft_fp16.sh)


# 6. Prompt tech
- [persona assignment](prompt/readme.md#1-persona-assignment)


# 7. datasketch: lsh for data process, dup-remove
- [Keep Chinese, English and underline, no punctuation](dataprocess/remove_none_ch_en.py)
- [minhash lsh for similarity](dataprocess/minhash_similarity.py)
- [lsh cluster](https://github.com/ybdesire/llm_learning/blob/main/dataprocess/datasketch_lsh_cluster.py)

# 8. fastparquet
- [read parquet file](dataprocess/fastparquet/read_parquet_file.py)

# 9. dataset construct
- [prompt prefix for sft dataset](dataprocess/prompt_prefix_en.py)

# 10. FlagEmbedding 
- [BGE-M3 embedding for text](flag_embedding/BGEM3.py)

# 11. BERT
- [bert for long text classification](bert/bert_for_long_text_classification.py)

# 12. datasets
- [load and show data structure](datasets/load_parse_cc_news.py)

# 13. datasketch
- [remove duplicated text by minhash for data preprocessing](datasketch/dup_text_remove_by_minhash.py)


