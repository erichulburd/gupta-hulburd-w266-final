import tensorflow as tf
from bert.modeling import BertConfig, get_shape_list


class CNNConfig:
    #filter_shapes: [[int]]
    #pool_shapes: [[int]]
    #channels_out: [int]
    #max_seq_length: int
    #bert_config: BertConfig

    def __init__(self, filter_shapes, pool_shapes, channels_out, max_seq_length, bert_config,
                 model):
        self.filter_shapes = filter_shapes
        self.pool_shapes = pool_shapes
        self.channels_out = channels_out
        self.max_seq_length = max_seq_length
        self.bert_config = bert_config
        self.model = model

    def serialize(self):
        return {
            'filter_shapes': self.filter_shapes,
            'pool_shapes': self.pool_shapes,
            'channels_out': self.channels_out,
            'max_seq_length': self.max_seq_length,
            'bert_config': self.bert_config.to_dict(),
            'model': self.model,
        }


def _conv_layer(layer, filter_shape, pool_shape, channels_in, channels_out, name):
    seq_length = layer.shape[1]
    hidden_size = layer.shape[2]

    # setup the filter input shape for tf.nn.conv_2d
    assert filter_shape[1] == hidden_size
    assert channels_in == 1
    assert filter_shape[0] <= seq_length
    conv_filt_shape = [filter_shape[0], filter_shape[1], channels_in, channels_out]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
    bias = tf.Variable(tf.truncated_normal([channels_out]), name=name + '_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(layer, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 1, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def create_cnn_model(is_training, token_embeddings, config: CNNConfig):
    """Creates a classification model."""

    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]

    channels_in = 1

    conv = tf.reshape(token_embeddings, [batch_size, seq_length, hidden_size, channels_in])
    for i, filter_shape in enumerate(config.filter_shapes):
        pool_shape = config.pool_shapes[i]
        conv = _conv_layer(conv, filter_shape, pool_shape, channels_in, config.channels_out[i],
                           ('convfilter%d' % i))

    w_out = (conv.shape[-1] * conv.shape[-2]).value

    n_positions = 2  # start and end logits
    wd1 = tf.Variable(tf.truncated_normal([w_out, n_positions], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_positions], stddev=0.01), name='bd1')

    conv = tf.reshape(conv, [batch_size * seq_length, w_out])
    logits = tf.matmul(conv, wd1)
    logits = tf.nn.bias_add(logits, bd1)

    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
