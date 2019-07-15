import tensorflow as tf
from bert.modeling import BertConfig, get_shape_list


class LSTMConfig:
    def __init__(self, num_units: int, max_seq_length: int, model: str, bert_config: BertConfig):
        self.num_units = num_units
        self.max_seq_length = max_seq_length
        self.bert_config = bert_config
        assert model == 'lstm'
        self.model = model

    def serialize(self):
        return {
            'num_units': self.num_units,
            'max_seq_length': self.max_seq_length,
            'bert_config': self.bert_config.to_dict(),
            'model': self.model
        }


def create_rnn_lstm_model(is_training, token_embeddings, config):
    # for example https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    n_positions = 2
    out_weights = tf.Variable(tf.random_normal([config.num_units, seq_length * n_positions]))
    out_bias = tf.Variable(tf.random_normal([seq_length * n_positions]))

    inpt = tf.unstack(token_embeddings, seq_length, 1)

    lstm_layer = tf.compat.v1.nn.rnn_cell.LSTMCell(config.num_units, forget_bias=1)
    outputs, _ = tf.nn.static_rnn(lstm_layer, inpt, dtype="float32")

    logits = tf.matmul(outputs[-1], out_weights)
    logits = tf.nn.bias_add(logits, out_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
