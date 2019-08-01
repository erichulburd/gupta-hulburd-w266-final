import tensorflow as tf
from bert.modeling import get_shape_list
from utils import mask_questions_batch


class FullyConnectedConfig:
    def __init__(self, max_seq_length, bert_config, mask_questions=False, model='fully_connected'):
        self.max_seq_length = max_seq_length
        self.mask_questions = mask_questions
        self.bert_config = bert_config
        self.model = model

    def serialize(self):
        return {
            'max_seq_length': self.max_seq_length,
            'mask_questions': self.mask_questions,
            'bert_config': self.bert_config.to_dict(),
            'model': self.model,
        }


def create_fully_connected_model(is_training, token_embeddings, config, batch_size, segment_ids):
    """Creates a classification model."""
    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    # batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]

    channels_in = 1

    n_positions = 2  # start and end logits

    output_weights = tf.get_variable("cls/squad/output_weights", [2, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    if config.mask_questions:
        token_embeddings = mask_questions_batch(token_embeddings, segment_ids, hidden_size)

    final_hidden_matrix = tf.reshape(token_embeddings, [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)
