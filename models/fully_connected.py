import tensorflow as tf
from bert.modeling import get_shape_list


def _create_fully_connected_model(is_training, token_embeddings):
    """Creates a classification model."""
    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]

    channels_in = 1

    n_positions = 2  # start and end logits
    wd1 = tf.Variable(tf.truncated_normal([hidden_size * channels_in, n_positions], stddev=0.03),
                      name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_positions], stddev=0.01), name='bd1')

    token_embeddings = tf.reshape(token_embeddings,
                                  [batch_size * seq_length, hidden_size * channels_in])
    logits = tf.matmul(token_embeddings, wd1)
    logits = tf.nn.bias_add(logits, bd1)

    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
