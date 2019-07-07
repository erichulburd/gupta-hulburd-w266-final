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
        example['paragraphs'] = mask_questions(example['token_embeddings'], example['segment_ids'],
                                               bert_config.hidden_size)
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


def mask_questions(token_embeddings, segment_ids, hidden_depth):
    full_segment_ids = tf.cast(segment_ids, tf.float32)
    full_segment_ids = tf.expand_dims(full_segment_ids, axis=1)
    full_segment_ids = tf.keras.backend.repeat_elements(full_segment_ids, hidden_depth, axis=1)
    return tf.multiply(token_embeddings, full_segment_ids)


def mask_questions_batch(token_embeddings, segment_ids, hidden_depth):
    full_segment_ids = tf.cast(segment_ids, tf.float32)
    full_segment_ids = tf.expand_dims(full_segment_ids, axis=2)
    full_segment_ids = tf.keras.backend.repeat_elements(full_segment_ids, hidden_depth, axis=2)
    return tf.multiply(token_embeddings, full_segment_ids)


def compute_batch_accuracy(logits, positions):
    predictions = tf.arg_max(logits, 1)
    correct_predictions = tf.cast(tf.equal(tf.cast(predictions, tf.int32), positions), tf.int32)
    return tf.math.divide(tf.reduce_sum(correct_predictions), positions.get_shape()[0])


def compute_weighted_batch_accuracy(logits, positions, k):
    _, top_indices = tf.math.top_k(logits, k=k, sorted=True)

    def _calc_accuracies(i):
        multiplier = (tf.constant(k, dtype=tf.float64) - tf.cast(i, tf.float64)) / tf.constant(
            k, dtype=tf.float64)
        predictions = top_indices[:, tf.cast(i, tf.int32)]
        correct_predictions = tf.cast(tf.equal(tf.cast(predictions, tf.int32), positions), tf.int32)

        return multiplier * tf.math.divide(tf.reduce_sum(correct_predictions),
                                           positions.get_shape()[0])

    return tf.reduce_sum(tf.map_fn(_calc_accuracies, tf.constant(list(range(k)), tf.float64)))
