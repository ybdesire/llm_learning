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
    --model_name_or_path THUDM/chatglm2-6b \ # base model path
    --output_dir output/my-1k-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \ # output tuning model checkpoint path
    --overwrite_output_dir \
    --max_source_length 3072 \ # max input text length
    --max_target_length 3072 \ # max output text length
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 5000 \ # like epoch
    --logging_steps 10 \ # ouput log each 10 steps
    --save_steps 500 \ # save a mid model each 500 steps
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 
