import tensorflow as tf
from bert.modeling import BertConfig, get_shape_list
from utils import mask_questions_batch
from models.cnn_keras import CNNKerasConfig, apply_conv_layers


class ContextualizedCNNConfig:
    """
    max_seq_length: int
    filter_generator_pooling: dict
    filter_generator: dict
    cnn_downsize: dict
    bert_cong: BertConfig
    """

    def __init__(self,
                 max_seq_length,
                 cnn_downsize,
                 filter_generator,
                 filter_generator_pooling,
                 bert_config,
                 mask_questions=True,
                 model='contextualized_cnn'):
        self.max_seq_length = max_seq_length
        cnn_downsize['bert_config'] = bert_config
        cnn_downsize['max_seq_length'] = max_seq_length
        self.cnn_downsize = CNNKerasConfig(**cnn_downsize)
        filter_generator['bert_config'] = bert_config
        filter_generator['max_seq_length'] = max_seq_length
        self.filter_generator = CNNKerasConfig(**filter_generator)
        self.filter_generator_pooling = filter_generator_pooling
        self.bert_config = bert_config
        self.mask_questions = mask_questions
        self.model = model

        self._validate()

    def _validate(self):
        assert self.cnn_downsize.channels_out[-1] <= self.filter_generator.channels_out[0]
        assert (self.filter_generator.channels_out[0] % self.cnn_downsize.channels_out[-1]) == 0

    def serialize(self):
        return {
            'max_seq_length': self.max_seq_length,
            'cnn_downsize': self.cnn_downsize.serialize(),
            'filter_generator': self.filter_generator.serialize(),
            'filter_generator_pooling': self.filter_generator_pooling,
            'bert_config': self.bert_config.to_dict(),
            'mask_questions': self.mask_questions,
            'model': self.model
        }


def apply_per_sample_conv1d(a, filters, n_filters):
    batch_size = a.shape[0].value
    seq_length = a.shape[1].value
    channels_in = a.shape[2].value

    a = tf.transpose(a, [1, 0, 2])
    a = tf.reshape(a, [1, seq_length, 1, batch_size * channels_in])

    assert filters.shape[0].value == batch_size
    filter_length = filters.shape[1].value
    assert filter_length <= seq_length
    assert filters.shape[2].value == channels_in * n_filters

    filters = tf.reshape(filters, [batch_size, filter_length, channels_in, n_filters])

    filters = tf.transpose(filters, [1, 0, 3, 2])
    filters = tf.reshape(filters, [filter_length, 1, channels_in * batch_size, n_filters])

    result = tf.nn.depthwise_conv2d(a, filters, strides=[1, 1, 1, 1], padding='SAME')
    result = tf.reshape(result, [seq_length, batch_size, channels_in, n_filters])
    result = tf.transpose(result, [1, 0, 2, 3])

    result = tf.reduce_sum(result, axis=2)
    return result


def create_contextualized_cnn_model(is_training,
                                    token_embeddings,
                                    config: ContextualizedCNNConfig,
                                    segment_ids=None,
                                    name="deconv_model"):
    """
    Impossible to train. I was getting examples/sec: 0.152973
    May need to take a closer look here and figure out if something is wrong
    or perhaps even implement this ourselves.
    https://arxiv.org/pdf/1603.07285.pdf
    """

    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    channels_in = 1

    downsized_input = token_embeddings
    if len(config.cnn_downsize.filter_shapes) > 0:
        downsized_input = apply_conv_layers(
            is_training,
            token_embeddings,
            config.cnn_downsize,
            dropout_rate=0.,
            name='contextualized_cnn/downsizer',
        )

    assert downsized_input.shape[0].value == batch_size
    assert downsized_input.shape[1].value == seq_length
    downsized_channels_out = downsized_input.shape[-1].value

    paragraphs = downsized_input
    if config.mask_questions:
        paragraphs = mask_questions_batch(downsized_input, segment_ids, downsized_channels_out)

    filters = apply_conv_layers(is_training,
                                downsized_input,
                                config.filter_generator,
                                name='contextualized_cnn/filter_generator',
                                dropout_rate=0.)

    assert filters.shape[0].value == batch_size
    assert filters.shape[1].value == seq_length
    n_filters = int(filters.shape[2].value / downsized_channels_out)

    filter_generator_pooling = config.filter_generator_pooling
    pooling_size = [filter_generator_pooling['size'], 1, 1]
    pooling_strides = [filter_generator_pooling['strides'], 1, 1]
    if config.filter_generator_pooling['type'] == 'max':
        filters = tf.nn.max_pool1d(filters, pooling_size, pooling_strides, 'VALID')
    elif config.filter_generator_pooling['type'] == 'avg':
        filters = tf.nn.avg_pool1d(filters, pooling_size, pooling_strides, 'VALID')

    contextualized = apply_per_sample_conv1d(paragraphs, filters, n_filters)

    n_positions = 2  # start and end logits
    wd1 = tf.Variable(tf.truncated_normal([n_filters, n_positions], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_positions], stddev=0.01), name='bd1')

    logits = tf.matmul(contextualized, wd1)
    logits = tf.nn.bias_add(logits, bd1)

    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
