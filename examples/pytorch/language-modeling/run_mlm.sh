# # Train bert-base on wiki103 lr=1e-4 epoch=10
# WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=2 nohup /bin/python3 run_mlm.py \
#     --model_type=bert --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 --tokenizer_name=bert-base-uncased \
#     --per_device_train_batch_size=128 --per_device_eval_batch_size=1024 --do_train --do_eval --fp16 \
#     --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --overwrite_output_dir --include_num_input_tokens_seen \
#     --warmup_ratio=0.2 --weight_decay=0.01 --max_seq_length=128 --learning_rate=1e-4 --num_train_epochs=10 \
#     --output_dir=/results/hf_pretrain/bert-pt-wiki103-fp16-1e-4-seq-128-epoch-10 > /results/hf_pretrain/bert-pt-wiki103-fp16-1e-4-seq-128-epoch-10.out 2>&1 &


# # Train bert-base on wiki103 lr=5e-5 epoch=10 warmup=0.1
# WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=2 nohup /bin/python3 run_mlm.py \
#     --model_type=bert --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 --tokenizer_name=bert-base-uncased \
#     --per_device_train_batch_size=128 --per_device_eval_batch_size=1024 --do_train --do_eval --fp16 \
#     --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --overwrite_output_dir --include_num_input_tokens_seen \
#     --warmup_ratio=0.1 --weight_decay=0.01 --max_seq_length=128 --learning_rate=5e-5 --num_train_epochs=10 \
#     --output_dir=/results/hf_pretrain/bert-pt-wiki103-fp16-5e-5-seq-128-epoch-10-warmup-0.1 \
#     > /results/hf_pretrain/bert-pt-wiki103-fp16-5e-5-seq-128-epoch-10-warmup-0.1.out 2>&1 &


# # Fine-tune roberta-base on wiki103 lr=5e-5 epoch=3
# WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=0 nohup /bin/python3 run_mlm.py \
#     --model_name_or_path=roberta-base --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 \
#     --per_device_train_batch_size=64 --per_device_eval_batch_size=1024 --do_train --do_eval --fp16 \
#     --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --include_num_input_tokens_seen \
#     --warmup_ratio=0.1 --max_seq_length=128 --learning_rate=5e-5 --num_train_epochs=3 --overwrite_output_dir\
#     --output_dir=/results/hf_pretrain/roberta-ft-wiki103-fp16 > /results/hf_pretrain/roberta-ft-wiki103-fp16.out 2>&1 &


# # Fine-tune bert-base on wiki103 lr=5e-5 epoch=3
# WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=0 nohup /bin/python3 run_mlm.py \
#     --model_name_or_path=bert-base-uncased --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 \
#     --per_device_train_batch_size=64 --per_device_eval_batch_size=128 --do_train --do_eval --fp16 \
#     --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --include_num_input_tokens_seen \
#     --warmup_ratio=0.1 --max_seq_length=128 --learning_rate=5e-5 --num_train_epochs=3 --overwrite_output_dir\
#     --output_dir=/results/hf_pretrain/bert-ft-wiki103-fp16 > /results/hf_pretrain/bert-ft-wiki103-fp16.out 2>&1 &


# # Train tensor-bert-base on wiki103 lr=1e-4 epoch=10
# WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=0 nohup /bin/python3 run_mlm.py \
#     --model_type=nv_tensor_bert --tensor_backend --config_name=/workspace/transformers/nv_bert_config/base-tensor.json \
#     --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 --tokenizer_name=bert-base-uncased \
#     --per_device_train_batch_size=128 --per_device_eval_batch_size=1024 --do_train --do_eval --fp16 \
#     --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --overwrite_output_dir --include_num_input_tokens_seen \
#     --warmup_ratio=0.2 --weight_decay=0.01 --max_seq_length=128 --learning_rate=1e-4 --num_train_epochs=10 \
#     --output_dir=/results/hf_pretrain/tensor-bert-pt-wiki103-fp16-1e-4-seq-128-epoch-10 --save_safetensors=False \
#     > /results/hf_pretrain/tensor-bert-pt-wiki103-fp16-1e-4-seq-128-epoch-10.out 2>&1 &


# Train tensor-bert-base-moe on wiki103 lr=1e-4 epoch=10
CONFIG_NAME=base-tensor-moe-32-4-false-64
RUN_NAME=$CONFIG_NAME
WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=0 nohup /bin/python3 run_mlm.py \
    --model_type=nv_tensor_bert_moe --tensor_backend --config_name=/workspace/transformers/nv_bert_config/$CONFIG_NAME.json \
    --dataset_name=wikitext --dataset_config_name=wikitext-103-raw-v1 --tokenizer_name=bert-base-uncased \
    --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --do_train --do_eval --fp16 \
    --eval_strategy=epoch --save_strategy=epoch --logging_steps=10 --overwrite_output_dir --include_num_input_tokens_seen \
    --warmup_ratio=0.2 --weight_decay=0.01 --max_seq_length=128 --learning_rate=1e-4 --num_train_epochs=10 \
    --output_dir=/results/hf_pretrain/$RUN_NAME --save_safetensors=False \
    > /results/hf_pretrain/$RUN_NAME.out 2>&1 &