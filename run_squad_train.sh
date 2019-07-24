#!/bin/sh -f

BERT_LARGE_DIR='/Users/suhasgupta/w266/gupta-hulburd-w266-final/BERT_BASE_DIR' 
SQUAD_DIR='/Users/suhasgupta/w266/gupta-hulburd-w266-final/SQUAD_DATA' 
OUTPUT_DIR='/Users/suhasgupta/w266/gupta-hulburd-w266-final/out' 

echo $BERT_LARGE_DIR 
echo $SQUAD_DIR
echo $OUTPUT_DIR

python3 bert/run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=32 \
  --max_query_length=24 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR \
  --use_tpu=False \
  --version_2_with_negative=True
