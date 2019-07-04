import tensorflow as tf

# NOTE: We are using the default value from the run_squad.py script.
MAX_SEQ_LENGTH = 384


def input_fn_builder(input_file, seq_length, is_training, drop_remainder, bert_config):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_embeddings": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            elif t.dtype == tf.float64:
                t = tf.cast(t, tf.float32)
            example[name] = t

        example['token_embeddings'] = tf.reshape(example['token_embeddings'],
                                                 [seq_length, bert_config.hidden_size])

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                          batch_size=batch_size,
                                          drop_remainder=drop_remainder))

        return d

    return input_fn


def flat_map_create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(
        [embedding_value for token_embeddings in values for embedding_value in token_embeddings])))
