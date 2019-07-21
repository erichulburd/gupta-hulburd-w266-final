import tensorflow as tf
from datetime import datetime
import json
from bert.modeling import get_assignment_map_from_checkpoint, get_shape_list, BertConfig
from bert.optimization import create_optimizer
from utils import (MAX_SEQ_LENGTH, input_fn_builder, compute_batch_accuracy,
                   compute_weighted_batch_accuracy)
from models.rnn_lstm import create_rnn_lstm_model, LSTMConfig
from models.cnn import CNNConfig, create_cnn_model
from models.contextualized_cnn import create_cnn_gan_model, CNNGANConfig
from utils import make_filename

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", MAX_SEQ_LENGTH, "The maximum total input sequence"
                     "length after WordPiece tokenization.")

flags.DEFINE_string("data_bert_directory", 'data/uncased_L-12_H-768_A-12',
                    'directory containing BERT config and checkpoints')

flags.DEFINE_string("init_checkpoint", None, '')

flags.DEFINE_string("output_dir", "out/%s" % datetime.now().isoformat(),
                    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 20, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_integer("eval_start_delay_secs", 120, "How many steps to make in each estimator call.")
flags.DEFINE_integer("eval_throttle_secs", 450, "How many steps to make in each estimator call.")
flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for prediction.")
# this should be N_eval / eval_batch_size
flags.DEFINE_integer("eval_steps", 20, "Number eval batches")

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

flags.DEFINE_integer("n_examples", None, "Name of tf_record file to use for input")
flags.DEFINE_float("eval_percent", 0.1,
                   "Percent of train set for evaluation (for finding filename).")
flags.DEFINE_string("config", "config.json", "JSON file with model configuration.")

DATA_BERT_DIRECTORY = FLAGS.data_bert_directory
BERT_CONFIG_FILE = "%s/bert_config.json" % DATA_BERT_DIRECTORY

OUTPUT_DIR = FLAGS.output_dir
INIT_CHECKPOINT = None
if FLAGS.init_checkpoint is not None:
    INIT_CHECKPOINT = '%s/%s' % (OUTPUT_DIR, FLAGS.init_checkpoint)

N_TRAIN_EXAMPLES = FLAGS.n_examples
TRAIN_FILE_NAME = make_filename('train', (1.0 - FLAGS.eval_percent), 'out/features',
                                N_TRAIN_EXAMPLES)
EVAL_FILE_NAME = make_filename('eval', (FLAGS.eval_percent), 'out/features', N_TRAIN_EXAMPLES)

tf.gfile.MakeDirs(OUTPUT_DIR)

bert_config = BertConfig.from_json_file(BERT_CONFIG_FILE)


def load_and_save_config(filename: str):
    with open(filename) as json_data:
        parsed = json.load(json_data)
        parsed['max_seq_length'] = FLAGS.max_seq_length
        parsed['bert_config'] = bert_config.to_dict()

        with open('%s/config.json' % OUTPUT_DIR, 'w') as f:
            json.dump(parsed, f)
        parsed['bert_config'] = bert_config

        create_model = None
        config_class = None
        if parsed['model'] == 'lstm':
            config_class = LSTMConfig
            create_model = create_rnn_lstm_model
        elif parsed['model'] == 'cnn':
            config_class = CNNConfig
            create_model = create_cnn_model
        elif parsed['model'] == 'cnn_gan':
            config_class = CNNGANConfig
            create_model = create_cnn_gan_model
        else:
            raise ValueError('No supported model %s' % parsed['model'])

        return (config_class(**parsed), create_model)


(config, create_model) = load_and_save_config(FLAGS.config)


def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps,
                     use_tpu, config):
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

        (start_logits, end_logits) = create_model(is_training,
                                                  token_embeddings,
                                                  config,
                                                  segment_ids=segment_ids)

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

        start_accuracy = compute_batch_accuracy(start_logits, start_positions)
        end_accuracy = compute_batch_accuracy(end_logits, end_positions)
        start_weighted5_accuracy = compute_weighted_batch_accuracy(start_logits,
                                                                   start_positions,
                                                                   k=5)
        end_weighted5_accuracy = compute_weighted_batch_accuracy(end_logits, end_positions, k=5)

        def write_summaries(family):
            tf.summary.scalar("start_loss", start_loss, family=family)
            tf.summary.scalar("end_loss", end_loss, family=family)
            tf.summary.scalar("total_loss", total_loss, family=family)

            tf.summary.scalar("start_accuracy", start_accuracy, family=family)
            tf.summary.scalar("end_accuracy", end_accuracy, family=family)
            tf.summary.scalar("start_weighted5_accuracy", start_weighted5_accuracy, family=family)
            tf.summary.scalar("end_weighted5_accuracy", end_weighted5_accuracy, family=family)

        if mode == tf.estimator.ModeKeys.TRAIN:
            write_summaries('train')

            # NOTE: We are using BERT's AdamWeightDecayOptimizer. We may want to reconsider
            # this if we are not able to effectively train.
            train_op = create_optimizer(total_loss, learning_rate, num_train_steps,
                                        num_warmup_steps, use_tpu)

            summaries = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=OUTPUT_DIR,
                summary_op=tf.compat.v1.summary.merge_all(),
            )
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          training_hooks=[summaries],
                                                          scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            write_summaries('eval')
            summaries = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=OUTPUT_DIR,
                summary_op=tf.compat.v1.summary.merge_all(),
            )
            predictions = {}

            # inference_model = initialize_inference_model(hypes)
            output_spec = tf.estimator.EstimatorSpec(mode,
                                                     loss=total_loss,
                                                     evaluation_hooks=[summaries],
                                                     predictions=predictions)

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
                                          log_step_count_steps=1,
                                          save_summary_steps=2,
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          keep_checkpoint_max=2,
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
                                config=config,
                                use_tpu=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                            model_fn=model_fn,
                                            config=run_config,
                                            train_batch_size=FLAGS.train_batch_size,
                                            eval_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.

        train_input_fn = input_fn_builder(input_file=TRAIN_FILE_NAME,
                                          seq_length=FLAGS.max_seq_length,
                                          is_training=True,
                                          bert_config=bert_config,
                                          drop_remainder=True)
        eval_input_fn = input_fn_builder(
            input_file=EVAL_FILE_NAME,
            seq_length=FLAGS.max_seq_length,
            # No need to shuffle eval set
            is_training=False,
            bert_config=bert_config,
            drop_remainder=True)
        # This should be .train_and_evaluate
        # https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
        # and https://towardsdatascience.com/how-to-configure-the-train-and-evaluate-loop-of-the-tensorflow-estimator-api-45c470f6f8d
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            # start_delay_secs=FLAGS.eval_start_delay_secs,  # start evaluating after N seconds
            throttle_secs=FLAGS.eval_throttle_secs,
            steps=FLAGS.eval_steps,
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()
