#!/bin/sh -f

#TPU_NAME="suhas-gupta"
MASTER_URL="grpc://10.0.101.2:8470"
TPU_ZONE="asia-east1-c"
GCP_PROJECT="w266"

 python3 train.py\
    --do_train=True\
    --config cnn_config.json\
    --tf_record train\
    --do_train=True\
    --num_train_examples=100\
    --num_train_epochs=1\
    --use_tpu=True\
    --tpu_zone=$TPU_ZONE\
    --master=$MASTER_URL\
    --gcp_project=$GCP_PROJECT
