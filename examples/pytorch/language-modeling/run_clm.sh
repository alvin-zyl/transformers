DEVICE=${DEVICE-"0"}
MODEL_TYPE=${MODEL_TYPE-"hf_tensor_gpt2"}

if [ "${MODEL_TYPE}" == "gpt2" ]; then
    readonly model_flag="--model_type=$MODEL_TYPE"
    readonly config_flag=""
    RUN_NAME=$MODEL_TYPE
else
    CONFIG_NAME=${CONFIG_NAME-"gpt2-tensor"}
    readonly model_flag="--model_type=$MODEL_TYPE --tensor_backend"
    readonly config_flag="--config_name=/workspace/transformers/hf_gpt2_configs/$CONFIG_NAME.json"
    RUN_NAME=$CONFIG_NAME
fi

TAG=${TAG-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi

DATA=${DATA-"allenai/c4"}
DATA_CONFIG=${DATA_CONFIG-"en"}
if [ "${DATA}" == "allenai/c4" ]; then
    readonly data_streaming_flag="--streaming"
    RUN_NAME=$RUN_NAME-C4
else
    readonly data_streaming_flag=""
fi

readonly data_flag="--dataset_name=$DATA --dataset_config_name=$DATA_CONFIG $data_streaming_flag"

LR=${LR-"2e-4"}
RUN_NAME=$RUN_NAME-LR-$LR

if [ "${MODEL_TYPE}" != "gpt2" ]; then
    TL=${TL-"none"}
    if [ "${TL}" == "none" ]; then
        readonly tensor_lr_flag=""
    else
        readonly tensor_lr_flag="--tensor_lr=$TL"
        RUN_NAME=$RUN_NAME-TL-$TL
    fi
fi

TENSOR_NO_DECAY=${TENSOR_NO_DECAY-"False"}
if [ "${TENSOR_NO_DECAY}" == "True" ]; then
    RUN_NAME=$RUN_NAME-TND
    readonly TND_flag="--tensor_no_decay=True"
else
    readonly TND_flag=""
fi

MGN=1.0
if [ "${MGN}" != "1.0" ]; then
    RUN_NAME=$RUN_NAME-MGN-$MGN
    readonly MGN_flag="--max_grad_norm=$MGN"
else
    readonly MGN_flag=""
fi

PRECESION=${PRECESION-"bf16"}
if [ "${PRECESION}" == "bf16" ]; then
    RUN_NAME=$RUN_NAME-$PRECESION
    readonly p_flag="--bf16"
elif [ "${PRECESION}" == "fp16" ]; then
    RUN_NAME=$RUN_NAME-$PRECESION
    readonly p_flag="--fp16"
else
    readonly p_flag=""
fi

BZ=${BZ-"32"}
GRAD_ACC=${GRAD_ACC-"4"}

OVERWRITE=${OVERWRITE-"False"}
if [ "${OVERWRITE}" == "True" ]; then
    readonly overwrite_flag="--overwrite_output_dir"
else
    readonly overwrite_flag=""
fi
MAX_STEP=${MAX_STEP-"18000"}

OPTIM=${OPTIM-"none"}
if [ "${OPTIM}" != "none" ]; then
    readonly optim_flag="--optim=$OPTIM"
else
    readonly optim_flag=""
fi

WANDB_PROJECT=hf_pretrain CUDA_VISIBLE_DEVICES=$DEVICE nohup /bin/python3 run_clm.py \
    $model_flag $config_flag $data_flag $optim_flag\
    --tokenizer_name=gpt2 \
    --per_device_train_batch_size=$BZ --per_device_eval_batch_size=$BZ --gradient_accumulation_steps=$GRAD_ACC \
    --do_train --do_eval $p_flag \
    --eval_strategy=steps --eval_steps=10000 --save_strategy=steps --save_steps=10000 --max_steps=$MAX_STEP \
    --logging_steps=10 --include_num_input_tokens_seen \
    --warmup_ratio=0.2 --weight_decay=0.01 \
    --learning_rate=$LR $tensor_lr_flag $TND_flag $MGN_flag \
    --output_dir=/results/hf_pretrain/$RUN_NAME --save_safetensors=False $overwrite_flag \
    > /results/hf_pretrain/$RUN_NAME.out 2>&1 &