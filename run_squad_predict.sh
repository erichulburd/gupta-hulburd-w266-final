#!/bin/sh -f

BERT_DIR='/home/suhasgupta/w266/gupta-hulburd-w266-final/data/uncased_L-12_H-768_A-12' 
SQUAD_DIR='/home/suhasgupta/w266/gupta-hulburd-w266-final/data/squad' 
OUTPUT_DIR='/home/suhasgupta/w266/gupta-hulburd-w266-final/out/run_squad_bert/' 

python3 bert/run_squad.py \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --do_train=False \
  --do_predict=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=32 \
  --max_query_length=24 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=False \
  --version_2_with_negative=True\
  --n_examples=1000
