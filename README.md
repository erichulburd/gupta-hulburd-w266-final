

# Data

## Squad

* Download dataset and evaluation script at [Squad explorer](https://rajpurkar.github.io/SQuAD-explorer/)
    * Place JSON data in data/squad/
    * Evaluation script goes in model/

## BERT

* Download pre-trained BERT embeddings [here](https://github.com/google-research/bert#pre-trained-models).
    * Currently we are using BERT-base uncased

## Processed Features

* We use `write_features.py` to save the Squad examples as contextualized BERT vectors.
* We use default values from the BERT `run_squad.py` file.
* The resulting `.tf_record` files are available on Google drive: https://drive.google.com/drive/u/2/folders/15RyBSMl0IlSUPW1_euanYkdYjOGNeQZO
    * Place them in `out/features`, so they can be read consistently across scripts.
