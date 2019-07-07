import tensorflow as tf
from utils import compute_batch_accuracy, compute_weighted_batch_accuracy, mask_questions_batch


def test_mask_questions():
    # batch_size = 2
    # seq_length = 3
    hidden_size = 4

    token_embeddings = tf.constant([[[0.99309865, 0.19019238, 0.990875, 0.72608557],
                                     [0.66810787, 0.3308506, 0.63321125, 0.22736391],
                                     [0.93261767, 0.70101261, 0.02638544, 0.67650275]],
                                    [[0.01551556, 0.23358984, 0.78789598, 0.28140234],
                                     [0.94942975, 0.59315012, 0.43632866, 0.6754952],
                                     [0.6390451, 0.07035357, 0.94329129, 0.30282875]]],
                                   dtype=tf.float32)

    segment_ids = tf.constant([[0, 0, 1], [0, 1, 1]])
    sess = tf.Session()
    with sess.as_default():
        paragraphs = mask_questions_batch(token_embeddings, segment_ids, hidden_size)
        assert paragraphs.eval().tolist() == [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
             [0.9326176643371582, 0.7010126113891602, 0.026385439559817314, 0.6765027642250061]],
            [[0.0, 0.0, 0.0, 0.0],
             [0.9494297504425049, 0.5931501388549805, 0.436328649520874, 0.6754952073097229],
             [0.6390451192855835, 0.07035356760025024, 0.9432913064956665, 0.30282875895500183]]
        ]
        sess.close()


def test_accuracy():
    positions = tf.constant([2, 0, 7])
    logits = tf.nn.softmax(
        tf.constant([
            [0., 0.1, 0.6, 0.3, 0.4, 0.5, 0.2, 0.7, 0.8, 0.9],
            [0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0., 0.9],
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.8, 0.7],
        ]))

    with tf.Session():
        assert compute_batch_accuracy(logits, positions).eval() == 1 / 3.
        expected_weighted_accuracy = 5 / 5. * 1 / 3. + 4 / 5. * 1 / 3. + 2 / 5. * 1 / 3.
        calculated_weighted_accuracy = compute_weighted_batch_accuracy(logits, positions,
                                                                       k=5).eval()
        assert abs(expected_weighted_accuracy - calculated_weighted_accuracy) < 10e-5


if __name__ == '__main__':
    test_accuracy()
    test_mask_questions()
