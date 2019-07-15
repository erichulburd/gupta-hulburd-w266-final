import tensorflow as tf

from bert.modeling import BertConfig, get_shape_list


class CNNGANConfig:
    depth: int
    channels_out: int
    max_seq_length: int
    bert_cong: BertConfig

    def __init__(self, depth: int, channels_out: int, max_seq_length: int, bert_config: BertConfig):
        self.depth = depth
        self.channels_out = channels_out
        self.max_seq_length = max_seq_length
        self.bert_config = bert_config

    def serialize(self):
        return {
            'depth': self.depth,
            'channels_out': self.channels_out,
            'max_seq_length': self.max_seq_length,
            'bert_config': self.bert_config.to_dict(),
        }


class PerSampleCNN:
    """
    Inspired by
    https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch#42086729
    """

    def __init__(
            self,
            # shape (batch_size, seq_length, hidden_depth, channels_in)
            inpt: tf.Tensor,
            # shape (batch_size, filter_height, filter_width, channels_in, channels_out)
            filters: tf.Tensor):
        self.input = inpt
        self.filters = filters

    def _validate(self):
        assert len(self.input.shape) == 4
        assert len(self.filters.shape) == 5
        # batch_size should match
        assert self.input.shape[0] == self.filters.shape[0]
        # sequence length should be gte than filter height
        assert self.input.shape[1] >= self.filters.shape[1]
        # input hidden depth should equal filter width in NLP context
        assert self.input.shape[2] == self.filters.shape[2]
        # channels in should match.
        assert self.input.shape[3] == self.filters.shape[3]

    def apply(self):
        self._validate()

        batch_size = self.input.shape[0]
        seq_length = self.input.shape[1]
        filter_height = self.filters.shape[1]
        width = self.input.shape[2]
        channels_in = self.input.shape[3]
        channels_out = self.filters.shape[4]

        virtual_channels_in = batch_size * channels_in

        filters = tf.transpose(self.filters, [1, 2, 0, 3, 4])
        filters = tf.reshape(filters, [filter_height, width, virtual_channels_in, channels_out])

        inpt = tf.transpose(self.input, [1, 2, 0, 3])  # shape (H, W, MB, channels_in)
        inpt = tf.reshape(inpt, [1, seq_length, width, virtual_channels_in])

        out = tf.nn.depthwise_conv2d(inpt, filter=filters, strides=[1, 1, 1, 1], padding="SAME")

        out = tf.reshape(out, [seq_length, width, batch_size, channels_in, channels_out])

        out = tf.transpose(out, [2, 0, 1, 3, 4])
        # sum over channels_in
        out = tf.reduce_sum(out, axis=3)
        # sum over width (ie hidden_size)
        return tf.reduce_sum(out, axis=2)


def create_deconv_model(bert_config, is_training, token_embeddings, paragraphs,
                        config: CNNGANConfig):
    """
    Impossible to train. I was getting examples/sec: 0.152973
    May need to take a closer look here and figure out if something is wrong
    or perhaps even implement this ourselves.
    https://arxiv.org/pdf/1603.07285.pdf
    """

    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    assert config.depth <= seq_length
    hidden_size = input_shape[2]

    channels_in = 1
    channels_out = config.channels_out

    token_embeddings = tf.reshape(token_embeddings,
                                  [batch_size, seq_length, hidden_size, channels_in])
    deconv_filter = tf.get_variable('cls/deconv/filter',
                                    [3, hidden_size, channels_out, channels_in],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_shape = tf.constant([batch_size, config.depth, hidden_size, channels_out],
                               dtype=tf.int32)
    # See https://github.com/vdumoulin/conv_arithmetic for a good explanation
    deconv = tf.nn.conv2d_transpose(
        token_embeddings,
        filter=deconv_filter,
        output_shape=output_shape,
        strides=[1, 1, 1, 1],
        # dilations=[1, 1, 1, 1],
        data_format='NHWC',
        padding='SAME',
        name='deconv')

    deconv = tf.reshape(deconv, [batch_size, config.depth, hidden_size, channels_in, channels_out])

    paragraphs = tf.reshape(paragraphs, [batch_size, seq_length, hidden_size, channels_in])
    conv = PerSampleCNN(inpt=paragraphs, filters=deconv).apply()

    n_positions = 2  # start and end logits
    wd1 = tf.Variable(tf.truncated_normal([channels_out, n_positions], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_positions], stddev=0.01), name='bd1')

    logits = tf.matmul(conv, wd1)
    logits = tf.nn.bias_add(logits, bd1)

    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
