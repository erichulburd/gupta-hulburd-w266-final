BERT_LARGE_DIR="gs://suhas-gupta-bucket/BERT_BASE_DIR"
SQUAD_DIR="gs://suhas-gupta-bucket/SQUAD_DIR"
OUT_DIR="gs://suhas-gupta-bucket/out"
TPU_NAME="suhas-gupta"


python bert/run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True

