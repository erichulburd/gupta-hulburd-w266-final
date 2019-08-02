# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
import pickle

from bert import modeling, tokenization
import tensorflow as tf
from utils import make_filename, read_squad_examples, convert_examples_to_features, FeatureWriter

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_bert_directory", 'data/uncased_L-12_H-768_A-12',
                    'directory containing BERT config and checkpoints')
flags.DEFINE_string("data_squad_directory", 'data/squad',
                    'directory containing raw Squad 2.0 examples')

# Other parameters
flags.DEFINE_bool(
    "do_lower_case", True, "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "fine_tune", False, "Whether to write SQUAD BERT embeddings to tf_record. "
    "Otherwise, it will write raw SQUAD features.")

flags.DEFINE_integer(
    "max_seq_length", 384, "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128, "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64, "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("write_dev", False, "Whether to also write test features.")

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

flags.DEFINE_integer("n_examples", None,
                     "Pass an integer here to limit the number of examples to save as features.")

flags.DEFINE_integer("num_tpu_cores", 8,
                     "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("version_2_with_negative", True,
                  "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_integer("batch_size", 32, "Total batch size.")

flags.DEFINE_float("eval_percent", 0.1, "Percent of training to set aside for validation")

flags.DEFINE_string("output_dir", 'out/features/',
                    "The output directory where the model checkpoints will be written.")

DATA_BERT_DIRECTORY = FLAGS.data_bert_directory
DATA_SQUAD_DIRECTORY = FLAGS.data_squad_directory

BERT_CONFIG_FILE = "%s/bert_config.json" % DATA_BERT_DIRECTORY
VOCAB_FILE = "%s/vocab.txt" % DATA_BERT_DIRECTORY
TRAIN_FILE = '%s/train-v2.0.json' % DATA_SQUAD_DIRECTORY
DEV_FILE = '%s/dev-v2.0.json' % DATA_SQUAD_DIRECTORY
INIT_CHECKPOINT = "%s/bert_model.ckpt" % DATA_BERT_DIRECTORY


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, INIT_CHECKPOINT)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model "
                         "was only trained up to sequence length %d" %
                         (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError("The max_seq_length (%d) must be greater than max_query_length "
                         "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    writer_fn = None
    if FLAGS.fine_tune:
        writer_fn = write_squad_features
    else:
        writer_fn = write_bert_embeddings

    if FLAGS.write_dev:
        dev_filename = make_filename('dev', 1.0, FLAGS.output_dir, FLAGS.fine_tune,
                                     FLAGS.n_examples)
        writer_fn(DEV_FILE,
                  True, [dev_filename], [1.0],
                  writing_dev=FLAGS.write_dev,
                  max_examples=FLAGS.n_examples)
        return

    splits = [1. - FLAGS.eval_percent, FLAGS.eval_percent]
    set_names = ['train', 'eval']

    writer_fn(TRAIN_FILE, True, [
        make_filename(set_name, split, FLAGS.output_dir, FLAGS.fine_tune, FLAGS.n_examples)
        for set_name, split in zip(set_names, splits)
    ], splits, FLAGS.n_examples)


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        model = modeling.BertModel(config=bert_config,
                                   is_training=False,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
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

        predictions = {
            "unique_id": unique_ids,
            'sequence_output': model.get_sequence_output(),
        }

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                      predictions=predictions,
                                                      scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def bert_input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
            tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask":
            tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids":
            tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def _parse_squad_features(input_file,
                          is_training,
                          output_files,
                          splits,
                          writing_dev=False,
                          max_examples=None):

    # STEP 1: Tokenize inputs
    examples = read_squad_examples(input_file=input_file,
                                   is_training=is_training,
                                   max_examples=max_examples,
                                   writing_dev=writing_dev)
    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(examples)
    idx = 0
    assert sum(splits) == 1.0
    example_sets = []
    for split in splits:
        next_idx = int(idx + math.ceil(split * len(examples)))
        print('idx, next_idx')
        print(idx, next_idx)
        example_sets.append(examples[idx:next_idx])
        idx = next_idx
    del examples

    for output_file, example_set in zip(output_files, example_sets):
        tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE,
                                               do_lower_case=FLAGS.do_lower_case)

        yield (output_file, example_set,
               convert_examples_to_features(examples=example_set,
                                            tokenizer=tokenizer,
                                            max_seq_length=FLAGS.max_seq_length,
                                            doc_stride=FLAGS.doc_stride,
                                            max_query_length=FLAGS.max_query_length,
                                            is_training=is_training))


def write_squad_features(input_file,
                         is_training,
                         output_files,
                         splits,
                         writing_dev=False,
                         max_examples=None):
    for output_file, examples, features in _parse_squad_features(input_file, is_training,
                                                                 output_files, splits, writing_dev,
                                                                 max_examples):

        feature_list = []
        for feature in features:
            feature_list.append(feature)

        suffix = ''
        if FLAGS.fine_tune:
            suffix = '_fine_tune'

        if writing_dev:
            with tf.gfile.GFile('%s/dev_examples%s.pickle' % (FLAGS.output_dir, suffix),
                                'wb') as out_file:
                pickle.dump(examples, out_file)
            with tf.gfile.GFile('%s/dev_features%s.pickle' % (FLAGS.output_dir, suffix),
                                'wb') as out_file:
                pickle.dump(feature_list, out_file)

        writer = FeatureWriter(filename=output_file, is_training=is_training)
        for i, feature in enumerate(feature_list):
            writer.process_feature(feature)

            if i % 1000 == 0:
                print('%d examples processed' % i)

        writer.close()


def write_bert_embeddings(input_file,
                          is_training,
                          output_files,
                          splits,
                          writing_dev=False,
                          max_examples=None):
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    for output_file, examples, features in _parse_squad_features(input_file, is_training,
                                                                 output_files, splits, writing_dev,
                                                                 max_examples):

        unique_id_to_feature = {}
        feature_list = []
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature
            feature_list.append(feature)

        if writing_dev:
            suffix = ''
            if FLAGS.fine_tune:
                suffix = '_fine_tune'
            with tf.gfile.GFile('%s/dev_examples%s.pickle' % (FLAGS.output_dir, suffix),
                                'wb') as out_file:
                pickle.dump(examples, out_file)
            with tf.gfile.GFile('%s/dev_features%s.pickle' % (FLAGS.output_dir, suffix),
                                'wb') as out_file:
                pickle.dump(feature_list, out_file)

        # STEP 2: initialize BERT model to extract token embeddings.
        layer_indexes = [-1]

        model_fn = model_fn_builder(bert_config=bert_config,
                                    init_checkpoint=INIT_CHECKPOINT,
                                    layer_indexes=layer_indexes,
                                    use_tpu=FLAGS.use_tpu,
                                    use_one_hot_embeddings=FLAGS.use_tpu)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(master=FLAGS.master,
                                              tpu_config=tf.contrib.tpu.TPUConfig(
                                                  num_shards=FLAGS.num_tpu_cores,
                                                  per_host_input_for_training=is_per_host))

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu,
                                                model_fn=model_fn,
                                                config=run_config,
                                                predict_batch_size=FLAGS.batch_size,
                                                train_batch_size=FLAGS.batch_size)

        input_fn = bert_input_fn_builder(features=feature_list, seq_length=FLAGS.max_seq_length)

        # STEP 3: Process token embeddings and write as tf_record.
        writer = FeatureWriter(filename=output_file, is_training=is_training)
        tf.logging.info("***** Writing features *****")
        tf.logging.info("    Num split examples = %d", writer.num_features)

        ct = 0

        for result in estimator.predict(input_fn, yield_single_examples=True):
            ct += 1
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            writer.process_feature(feature, result["sequence_output"])

            if ct % 1000 == 0:
                print('%d examples processed' % ct)

        writer.close()


if __name__ == "__main__":
    tf.app.run()
