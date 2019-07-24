#!/bin/sh -f

MASTER_URL="grpc://10.0.101.2:8470"
TPU_ZONE="asia-east1-c"
GCP_PROJECT="w266-240206"

 python3 train.py \
    --do_train=True \
    --config=lstm_config.json \
    --tf_record='train_non_fine_tuned' \
    --num_train_examples=100000 \
    --num_train_epochs=2 \
