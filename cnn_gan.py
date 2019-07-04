import tensorflow as tf
from datetime import datetime
import json
from bert.modeling import get_assignment_map_from_checkpoint, get_shape_list, BertConfig
from bert.optimization import create_optimizer
from utils import MAX_SEQ_LENGTH, input_fn_builder

# [X] Read in data
# [X] pass through conv2d_transpose (output A)
# [X] pass output A over input data
# [X] compute logits
# [X] compute loss
# [ ] optimize and train

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", MAX_SEQ_LENGTH, "The maximum total input sequence"
                     "length after WordPiece tokenization.")

flags.DEFINE_string("data_bert_directory", 'data/uncased_L-12_H-768_A-12',
                    'directory containing BERT config and checkpoints')

flags.DEFINE_string("init_checkpoint", None, '')

flags.DEFINE_string("output_dir", "out/features/%s" % datetime.now().isoformat(),
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for prediction.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("tf_record", "train", "Name of tf_record file to use for input")

DATA_BERT_DIRECTORY = FLAGS.data_bert_directory
BERT_CONFIG_FILE = "%s/bert_config.json" % DATA_BERT_DIRECTORY

OUTPUT_DIR = FLAGS.output_dir
INIT_CHECKPOINT = None
if FLAGS.init_checkpoint is not None:
    INIT_CHECKPOINT = '%s/%s' % (OUTPUT_DIR, FLAGS.init_checkpoint)

N_TRAIN_EXAMPLES = 1000


class CNNGANConfig:
    depth: int
    deconv_channels_out: int
    max_seq_length: int

    def __init__(self, depth: int, deconv_channels_out: int, max_seq_length: int):
        self.depth = depth
        self.deconv_channels_out = deconv_channels_out
        self.max_seq_length = max_seq_length


def make_cnn_gan_config(filename: str):
    with open(filename) as json_data:
        parsed = json.load(json_data)
        parsed['max_seq_length'] = FLAGS.max_seq_length
        return CNNGANConfig(**parsed)


bert_config = BertConfig.from_json_file(BERT_CONFIG_FILE)
cnn_gan_config = make_cnn_gan_config('cnn_gan_config.json')


class PerSampleCNN:
    """
    Inspired by https://stackoverflow.com/questions/42068999/tensorflow-convolutions-with-different-filter-for-each-sample-in-the-mini-batch#42086729
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
        # sum over width (ie hidden depth)
        return tf.reduce_sum(out, axis=2)


def _create_model(bert_config, is_training, token_embeddings, segment_ids,
                  cnn_gan_config: CNNGANConfig):
    """Creates a classification model."""

    input_shape = get_shape_list(token_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    assert cnn_gan_config.depth <= seq_length
    hidden_size = input_shape[2]

    channels_in = 1
    channels_out = cnn_gan_config.deconv_channels_out
    deconv_filter = tf.get_variable('cls/deconv/filter',
                                    [seq_length, hidden_size, channels_out, channels_in],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    # We want to set the input question values to 0, so they are not considered for
    # start and stop indices (segment_ids are 0 for question, 1 for answer).
    paragraphs = mask_questions(token_embeddings, segment_ids, hidden_size)

    token_embeddings = tf.reshape(token_embeddings,
                                  [batch_size, seq_length, hidden_size, channels_in])
    output_shape = tf.constant([batch_size, seq_length, hidden_size, channels_out], dtype=tf.int32)
    # See https://github.com/vdumoulin/conv_arithmetic for a good explanation
    deconv = tf.nn.conv2d_transpose(
        token_embeddings,
        filter=deconv_filter,
        output_shape=output_shape,
        strides=[1, 1, 1, 1],
        # dilations=[1, 1, 1, 1],
        data_format='NHWC',
        name='deconv')
    # Apply the deconvolutional layer over the paragraphs. Add bias add and RELU layer.
    ps_conv = PerSampleCNN(
        inpt=tf.reshape(paragraphs, [batch_size, seq_length, hidden_size, channels_in]),
        # F has shape (batch_size, filter_height, filter_width, channels_in, channels_out)
        filters=tf.reshape(deconv,
                           [batch_size, seq_length, hidden_size, channels_in, channels_out]))
    conv = ps_conv.apply()
    conv_bias = tf.get_variable("cls/squad/output_bias", [channels_out],
                                initializer=tf.zeros_initializer())
    conv = conv + conv_bias
    conv = tf.nn.relu(conv)

    n_positions = 2  # start and end logits
    wd1 = tf.Variable(tf.truncated_normal([channels_out, n_positions], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_positions], stddev=0.01), name='bd1')

    conv = tf.reshape(conv, [batch_size * seq_length, channels_out])

    logits = tf.matmul(conv, wd1)
    logits = tf.nn.bias_add(logits, bd1)

    print(logits.shape)
    logits = tf.reshape(logits, [batch_size, seq_length, n_positions])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def mask_questions(token_embeddings, segment_ids, hidden_depth):
    full_segment_ids = tf.cast(segment_ids, tf.float32)
    full_segment_ids = tf.expand_dims(full_segment_ids, axis=2)
    full_segment_ids = tf.keras.backend.repeat_elements(full_segment_ids, hidden_depth, axis=2)
    return tf.multiply(token_embeddings, full_segment_ids)


def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                     use_tpu, cnn_gan_config: CNNGANConfig):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        segment_ids = features["segment_ids"]
        token_embeddings = features["token_embeddings"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = _create_model(bert_config, is_training, token_embeddings,
                                                   segment_ids, cnn_gan_config)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            # NOTE: We are using BERT's AdamWeightDecayOptimizer. We may want to reconsider
            # this if we are not able to effectively train.
            train_op = create_optimizer(total_loss, learning_rate, num_train_steps,
                                        num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          predictions=predictions,
                                                          scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(OUTPUT_DIR)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_config = tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                          num_shards=FLAGS.num_tpu_cores,
                                          per_host_input_for_training=is_per_host)
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver,
                                          master=FLAGS.master,
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          tpu_config=tpu_config)

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_steps = int(N_TRAIN_EXAMPLES / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=INIT_CHECKPOINT,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                cnn_gan_config=cnn_gan_config,
                                use_tpu=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.train_batch_size,
                                            predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.

        train_input_fn = input_fn_builder(input_file="out/features/%s.tf_record" % FLAGS.tf_record,
                                          seq_length=FLAGS.max_seq_length,
                                          is_training=True,
                                          bert_config=bert_config,
                                          drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == "__main__":
    tf.app.run()
