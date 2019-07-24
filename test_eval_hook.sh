#!/bin/sh -f 

for eval_percent in 0.01 0.02 0.05 0.1 0.15 0.2
do
python3 process_squad_bert_embeddings.py --eval_percent $eval_percent
done
