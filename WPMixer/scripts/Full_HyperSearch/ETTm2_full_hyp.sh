#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Create WPMixer logs directory if it doesn't exist
if [ ! -d "./logs/WPMixer" ]; then
    mkdir ./logs/WPMixer
fi

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm2
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.00076587 0.000275775 0.000234608 0.001039536)
batches=(256 256 256 256)
wavelets=(bior3.1 db2 db2 db2)
levels=(1 1 1 1)
tfactors=(3 3 3 3)
dfactors=(8 7 5 8)
epochs=(80 80 80 80)
dropouts=(0.4 0.2 0.4 0.4)
embedding_dropouts=(0.0 0.1 0.0 0.0)
patch_lens=(48 48 48 48)
strides=(24 24 24 24)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 256)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	log_file="logs/${model_name}/full_hyperSearch_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
