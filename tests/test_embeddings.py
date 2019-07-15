import tensorflow as tf
from bert import modeling
from utils import input_fn_builder, MAX_SEQ_LENGTH

tf.enable_eager_execution()

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "max_seq_length", 384, "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("batch_size", 100, ".")

flags.DEFINE_string("data_bert_directory", 'data/uncased_L-12_H-768_A-12',
                    'directory containing BERT config and checkpoints')

bert_config = modeling.BertConfig.from_json_file("%s/bert_config.json" % FLAGS.data_bert_directory)

input_fn = input_fn_builder('out/features/eval.tf_record', FLAGS.max_seq_length, False, False,
                            bert_config)
dataset: tf.data.TFRecordDataset = input_fn({'batch_size': FLAGS.batch_size})


def test_embedding_dimensions():
    assert dataset.output_shapes['input_ids'].dims[0].value is None
    assert dataset.output_shapes['input_ids'].dims[1].value == FLAGS.max_seq_length
    assert len(dataset.output_shapes['input_ids'].dims) == 2

    assert dataset.output_shapes['input_mask'].dims[0].value is None
    assert dataset.output_shapes['input_mask'].dims[1].value == FLAGS.max_seq_length
    assert len(dataset.output_shapes['input_mask'].dims) == 2

    assert dataset.output_shapes['unique_ids'].dims[0].value is None
    assert len(dataset.output_shapes['unique_ids'].dims) == 1

    assert dataset.output_shapes['segment_ids'].dims[0].value is None
    assert dataset.output_shapes['segment_ids'].dims[1].value == FLAGS.max_seq_length
    assert len(dataset.output_shapes['segment_ids'].dims) == 2

    assert dataset.output_shapes['token_embeddings'].dims[0].value is None
    assert dataset.output_shapes['token_embeddings'].dims[1].value == FLAGS.max_seq_length
    assert dataset.output_shapes['token_embeddings'].dims[2].value == bert_config.hidden_size  # 768

    assert len(dataset.output_shapes['token_embeddings'].dims) == 3


def test_embedding_example():
    sess = tf.Session()
    with sess.as_default():
        for example in dataset.take(10):
            assert all(
                [len(segment_ids) == MAX_SEQ_LENGTH for segment_ids in example['segment_ids']])
            assert all([
                len(token_embeddings) == MAX_SEQ_LENGTH
                for token_embeddings in example['token_embeddings']
            ])
            assert all([
                len(embeddings) == bert_config.hidden_size
                for token_embeddings in example['token_embeddings']
                for embeddings in token_embeddings
            ])


if __name__ == '__main__':
    test_embedding_dimensions()
    test_embedding_example()
