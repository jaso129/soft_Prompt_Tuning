#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

POPSIZE=10
BUDGET=10
template=v1
initial=all

for dataset in asset
do
OUT_PATH=outputs/sim/$dataset/gpt/$initial/de/bd${BUDGET}_top${POPSIZE}_topk_para_init/$template/davinci
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --do_test \
    --dataset $dataset \
    --task sim \
    --batch-size 20 \
    --prompt-num 0 \
    --sample_num 100 \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode de \
    --llm_type davinci \
    --initial $initial \
    --initial_mode para_topk \
    --template $template \
    --cache_path data/sim/$dataset/seed$SEED/prompts_gpt.json \
    --output $OUT_PATH/seed${SEED}
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done