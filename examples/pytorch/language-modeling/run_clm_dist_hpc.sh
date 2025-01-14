#!/bin/bash
#SBATCH -A m4645_g
#SBATCH -N 1
#SBATCH -q premium
#SBATCH -C gpu
#SBATCH -t 1:00:00
#SBATCH --image=alvinliu12138/zhanggroup:dev_hf
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --volume="/global/cfs/cdirs/m4645/alvinliu/repo:/workspace;/global/cfs/cdirs/m4645/alvinliu/workspace/datasets:/datasets;/global/cfs/cdirs/m4645/alvinliu/workspace/results:/results"

PORT=$(($RANDOM + 10000))

TENSOR_MODEL_TYPE=("hf_tensor_gpt2", "hf_tensor_llama")
MODEL_TYPE=${MODEL_TYPE-"hf_tensor_gpt2"}

if [[ $(echo ${TENSOR_MODEL_TYPE[@]} | fgrep -w $MODEL_TYPE) ]]; then
    CONFIG_NAME=${CONFIG_NAME-"gpt2-tensor"}
    readonly model_flag="--model_type=$MODEL_TYPE --tensor_backend --config_name=/workspace/transformers/${MODEL_TYPE}_configs/$CONFIG_NAME.json"
    TOKENIZER=${TOKENIZER-"gpt2"}
    RUN_NAME=$CONFIG_NAME-$TOKENIZER-tokenizer
else
    CONFIG_NAME=${CONFIG_NAME-"none"}
    if [ "${CONFIG_NAME}" == "none" ]; then
        readonly model_flag="--model_type=$MODEL_TYPE"
        RUN_NAME=$MODEL_TYPE
    else
        readonly model_flag="--config_name=/workspace/transformers/${MODEL_TYPE}_configs/$CONFIG_NAME.json"
        RUN_NAME=$CONFIG_NAME
    fi
    TOKENIZER=${TOKENIZER-"$MODEL_TYPE"}
    if [ "${TOKENIZER}" != "$MODEL_TYPE" ]; then
        RUN_NAME=$RUN_NAME-$TOKENIZER-tokenizer
    fi
fi
readonly tokenizer_flag="--tokenizer_name=$TOKENIZER"

CONTINUE=${CONTINUE-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly checkpoint_flag="--model_name_or_path=$CONTINUE"
else
    readonly checkpoint_flag=""
fi

TRAIN=${TRAIN-"True"}
if [ "${TRAIN}" == "True" ]; then
    readonly train_flag="--do_train"
else
    readonly train_flag=""
fi
EVAL=${EVAL-$TRAIN}
if [ "${EVAL}" == "True" ]; then
    readonly eval_flag="--do_eval"
else
    readonly eval_flag=""
fi

TAG=${TAG-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi

HF_HOME="/datasets/.cache/huggingface"
DATA=${DATA-"/datasets/c4/tokenized"}
readonly data_flag="--dataset_name=$DATA --offline_mode"

LR=${LR-"2e-4"}
RUN_NAME=$RUN_NAME-LR-$LR

if [[ $(echo ${TENSOR_MODEL_TYPE[@]} | fgrep -w $MODEL_TYPE) ]]; then
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
MAX_STEP=${MAX_STEP-"22000"}

OPTIM=${OPTIM-"none"}
if [ "${OPTIM}" != "none" ]; then
    readonly optim_flag="--optim=$OPTIM"
else
    readonly optim_flag=""
fi

SEQ=${SEQ-"1024"}
readonly block_size_flag="--block_size=$SEQ"
if [ "${SEQ}" != "1024" ]; then
    RUN_NAME="$RUN_NAME-Seq-$SEQ"
fi

MIN_LR=${MIN_LR-"0.1"}
if [ "${MIN_LR}" == "none" ]; then
    readonly scheduler_flag="--lr_scheduler_type=cosine"
else
    readonly scheduler_flag='--lr_scheduler_type=cosine_with_min_lr --lr_scheduler_kwargs={"min_lr_rate":0.1}'
fi

NO_GROUP=${NO_GROUP-"True"}
if [ "${NO_GROUP}" == "True" ]; then
    readonly no_grouping_flag="--no_grouping --ignore_padding_tokens"
else
    readonly no_grouping_flag=""
    RUN_NAME=$RUN_NAME-GROUPED
fi

EVAL_STEPS=${EVAL_STEPS="11000"}
SAVE_STEPS=${SAVE_STEPS-$EVAL_STEPS}

if [ "${CONTINUE}" != "none" ]; then
    LOG_NAME=$RUN_NAME-${CONTINUE##*/}
else
    LOG_NAME=$RUN_NAME
fi

WANDB_PROJECT=hf_pretrain HF_HUB_OFFLINE=1 HF_HOME=$HF_HOME shifter torchrun --nproc-per-node=4 --master-port=$PORT run_clm.py \
    $model_flag $checkpoint_flag $data_flag $optim_flag $block_size_flag $tokenizer_flag \
    --per_device_train_batch_size=$BZ --per_device_eval_batch_size=$BZ --gradient_accumulation_steps=$GRAD_ACC \
    $train_flag $eval_flag $p_flag \
    --eval_strategy=steps --eval_steps=$EVAL_STEPS --save_strategy=steps --save_steps=$SAVE_STEPS --max_steps=$MAX_STEP \
    --logging_steps=10 --include_num_input_tokens_seen \
    --warmup_ratio=0.1 $scheduler_flag --weight_decay=0.01 \
    --learning_rate=$LR $tensor_lr_flag $TND_flag $MGN_flag \
    --output_dir=/results/hf_pretrain/$RUN_NAME --save_safetensors=False $overwrite_flag $no_grouping_flag
