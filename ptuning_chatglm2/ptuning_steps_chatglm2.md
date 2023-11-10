# ptuning steps chatglm2 by it's github repo code

1. `git clone https://github.com/THUDM/ChatGLM2-6B.git`
2. `cd ChatGLM2-6B/ptuning`
3. create data as format below ( `train.json` and `dev.json` )

```
{"prompt": "hello", "response": "hello", "history": []}
{"prompt": "hello1", "response": "hello1", "history": []}
{"prompt": "hello2", "response": "hello2", "history": []}
```

4. run shell script (below is one GPU with fp16, not int4)


```shell
PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1 # GPU Number

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file MyDataset/train.json \
    --validation_file MyDataset/dev.json \
    --preprocessing_num_workers 30 \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/my-1k-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 3072 \
    --max_target_length 3072 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 
```


5. check other .sh files at current folder for more different training methods 
