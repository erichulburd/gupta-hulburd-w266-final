import tensorflow as tf


class CNNKerasConfig:
    ''' (Type hints don't work with python3.5 which is avaialble with most cloud virtual machines)
    filter_shapes: [[int]]
    pool_shapes: [[int]]
    channels_out: [int]
    max_seq_length: int
    bert_config: BertConfig
    '''

    def __init__(self,
                 filter_shapes,
                 pool_shapes,
                 channels_out,
                 max_seq_length,
                 bert_config,
                 model='cnn'):
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


def _conv_layer(layer, filter_shape, channels_in, pool_shape, channels_out, name="conv_layer"):
    # setup the filter input shape for tf.nn.conv_2d
    assert channels_in == 1
    bias = tf.Variable(tf.truncated_normal([channels_out]), name=name + '_b')
    # setup the convolutional layer operation and Relu non-linearity
    out_layer = tf.keras.layers.Conv1D(channels_out,
                                       filter_shape[0],
                                       strides=1,
                                       padding='SAME',
                                       activation="relu")(layer)
    # add the bias
    out_layer += bias
    out_layer = tf.keras.layers.MaxPooling1D(pool_size=pool_shape, strides=1,
                                             padding="SAME")(out_layer)
    return out_layer


def apply_conv_layers(is_training,
                      token_embeddings,
                      config: CNNKerasConfig,
                      dropout_rate=0.2,
                      name="CNN"):
    dropout_rate = 0.2
    channels_in = 1  # Only one channel in input since we are doing NLP

    conv_all_layers_concatenated = []
    for i, filter_shape in enumerate(config.filter_shapes):
        pool_shape = config.pool_shapes[i]
        conv = _conv_layer(token_embeddings, filter_shape, channels_in, pool_shape,
                           config.channels_out[i], ('convfilter%d' % i))
        conv_all_layers_concatenated.append(conv)

    conv = None
    if len(conv_all_layers_concatenated) < 2:
        conv = conv_all_layers_concatenated[0]
    else:
        conv = tf.keras.layers.concatenate(conv_all_layers_concatenated, axis=2)

    if dropout_rate == 0.:
        return conv

    return tf.keras.layers.Dropout(rate=dropout_rate)(conv)


def create_cnnKeras_model(is_training,
                          token_embeddings,
                          config: CNNKerasConfig,
                          segment_ids=None,
                          name="CNN"):
    """Creates a classification model."""
    ############################ Layer 1 of CNNs ##############################
    conv1 = apply_conv_layers(is_training, token_embeddings, config, segment_ids, name)

    ############################ Layer 2 of CNNs ##############################
    conv2 = apply_conv_layers(is_training, conv1, config, segment_ids, name)

    ############################ Fully Connected ##############################
    n_positions = 2  # start and end logits
    logits = tf.keras.layers.Dense(n_positions, activation='softmax')(conv2)
    unstacked_logits = tf.unstack(logits, axis=2)
    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
    return (start_logits, end_logits)
