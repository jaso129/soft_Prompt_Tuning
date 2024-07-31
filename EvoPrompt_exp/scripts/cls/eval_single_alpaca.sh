set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

SEED=5
# TODO 評估提示分數

for dataset in agnews
do
python infer.py \
    --dataset $dataset \
    --seed $SEED \
    --position demon \
    --task cls \
    --batch-size 4 \
    --prompt-num 0 \
    --language_model  alpaca \
    --output outputs/cls/$dataset/eval/alpaca \
    --content  "Summarize the main event of the news article,
                Identify the primary topic of this news story,
                Classify the news into categories like sports, business, technology, or world,
                Highlight the key facts presented in this news piece,
                Determine if the news story contains any political content."
done