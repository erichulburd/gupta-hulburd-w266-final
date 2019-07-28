import tensorflow as tf
import numpy as np
from models.contextualized_cnn import apply_per_sample_conv1d


def _make_toy_example(batch_size, seq_length, filter_length, n_filters=1, channels_in=1):
    a_orig = np.reshape(np.array(list(range(batch_size * seq_length * channels_in))),
                        (batch_size, seq_length, channels_in))
    a = tf.constant(a_orig)
    a = tf.cast(a, tf.float64)

    filters_orig = np.reshape(
        np.array(list(range(batch_size * filter_length * channels_in * n_filters))),
        (batch_size, filter_length, channels_in * n_filters))
    filters = tf.constant(filters_orig)

    return (a, filters)


sess = tf.InteractiveSession()


def test1():
    n_filters = 1
    (a, filters) = _make_toy_example(5, 4, 2, n_filters=n_filters)
    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert result.shape[0] == 5
    assert result.shape[1] == 4
    assert result.shape[2] == n_filters


def test2():
    n_filters = 3
    (a, filters) = _make_toy_example(5, 4, 2, n_filters=n_filters)
    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert result.shape[0] == 5
    assert result.shape[1] == 4
    assert result.shape[2] == n_filters


def test3():
    batch_size = 1
    seq_length = 2
    channels_in = 1
    a = np.array(list(range(2)))
    a = np.reshape(a, (batch_size, seq_length, channels_in))
    a = tf.constant(a, tf.float64)

    filter_length = 1
    n_filters = 1
    filters = np.ones((batch_size, filter_length, channels_in * n_filters))
    filters = tf.constant(filters, tf.float64)

    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert (result == np.array([[[0.], [1.]]])).all()


def test4():
    batch_size = 3
    seq_length = 2
    channels_in = 1
    a = np.array(list(range(6)))
    a = np.reshape(a, (batch_size, seq_length, channels_in))
    a = tf.constant(a, tf.float64)

    filter_length = 1
    n_filters = 1
    filters = np.ones((batch_size, filter_length, channels_in * n_filters))
    filters = tf.constant(filters, tf.float64)

    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert (result == np.array([[[0.], [1.]], [[2.], [3.]], [[4.], [5.]]])).all()


def test5():
    batch_size = 3
    seq_length = 2
    channels_in = 2
    a = np.array(list(range(12)))
    a = np.reshape(a, (batch_size, seq_length, channels_in))
    a = tf.constant(a, tf.float64)

    filter_length = 1
    n_filters = 1
    filters = np.ones((batch_size, filter_length, channels_in * n_filters))
    filters = tf.constant(filters, tf.float64)

    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert (result == np.array([[[1.], [5.]], [[9.], [13.]], [[17.], [21.]]])).all()


def test6():
    batch_size = 4
    seq_length = 3
    channels_in = 2
    a = np.array(list(range(24)))
    a = np.reshape(a, (batch_size, seq_length, channels_in))
    a = tf.constant(a, tf.float64)

    filter_length = 1
    n_filters = 1
    filters = np.ones((batch_size, filter_length, channels_in * n_filters))
    filters = tf.constant(filters, tf.float64)

    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert (result == np.array([[[1.], [5.], [9.]], [[13.], [17.], [21.]], [[25.], [29.], [33.]],
                                [[37.], [41.], [45.]]])).all()


def test7():
    batch_size = 4
    seq_length = 3
    channels_in = 2
    a = np.array(list(range(24)))
    a = np.reshape(a, (batch_size, seq_length, channels_in))
    a = tf.constant(a, tf.float64)
    filter_length = 1
    n_filters = 2
    filters = np.ones((batch_size, filter_length, channels_in))
    filters2 = filters * 2
    filters = tf.constant(filters, tf.float64)
    filters2 = tf.constant(filters2, tf.float64)
    filters = tf.concat([filters, filters2], 2)

    result = apply_per_sample_conv1d(a, filters, n_filters)
    result = sess.run(result)
    assert (result == np.array([[[1., 2.], [5., 10.], [9., 18.]],
                                [[13., 26.], [17., 34.], [21., 42.]],
                                [[25., 50.], [29., 58.], [33., 66.]],
                                [[37., 74.], [41., 82.], [45., 90.]]])).all()


if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
