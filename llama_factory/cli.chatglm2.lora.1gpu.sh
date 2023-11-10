# command line demo to load chatglm2 lora fine tuning model

python src/cli_demo.py \
    --model_name_or_path /data1/yinbin/projects/LLaMAFactory/LLaMA-Factory/models/dataroot/models/THUDM/chatglm2-6b \
    --template chatglm2 \
    --finetuning_type lora \
    --checkpoint_dir output/chatglm2_sft_lora_my/checkpoint-50


