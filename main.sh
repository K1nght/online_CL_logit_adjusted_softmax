model=$1
dataset=$2
size=$3
tau=$4
l=$5

python ./utils/main.py --nowand 1 \
    --model=${model} --dataset=${dataset} \
    --buffer_size=${size} --n_epochs 1 --lr 0.03 \
    --minibatch_size 32 --batch_size 32 --tau ${tau} --window_length ${l}


