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

import collections
import json
import os
import random
import math

from bert import modeling, tokenization
import six
import tensorflow as tf
from utils import make_filename

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
flags.DEFINE_bool("write_test", False, "Whether to also write test features.")

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
TEST_FILE = '%s/dev-v2.0.json' % DATA_SQUAD_DIRECTORY
INIT_CHECKPOINT = "%s/bert_model.ckpt" % DATA_BERT_DIRECTORY


class SquadExample(object):
    """A single training/test example for simple sequence classification.

         For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, max_examples=None):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:

                    if FLAGS.version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                                               cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(qas_id=qas_id,
                                       question_text=question_text,
                                       doc_tokens=doc_tokens,
                                       orig_answer_text=orig_answer_text,
                                       start_position=start_position,
                                       end_position=end_position,
                                       is_impossible=is_impossible)
                examples.append(example)

                if max_examples is not None and len(examples) == max_examples:
                    return examples

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""
    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position,
             tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position,
                                                      tok_end_position, tokenizer,
                                                      example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)

            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" %
                                " ".join([tokenization.printable_text(x) for x in tokens]))
                tf.logging.info(
                    "token_to_orig_map: %s" %
                    " ".join(["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info(
                    "token_is_max_context: %s" %
                    " ".join(["%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(unique_id=unique_id,
                                    example_index=example_index,
                                    doc_span_index=doc_span_index,
                                    tokens=tokens,
                                    token_to_orig_map=token_to_orig_map,
                                    token_is_max_context=token_is_max_context,
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    start_position=start_position,
                                    end_position=end_position,
                                    is_impossible=example.is_impossible)

            unique_id += 1

            yield feature


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #     Question: What year was John Smith born?
    #     Context: The leader was John Smith (1895-1943).
    #     Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #     Question: What country is the top exporter of electornics?
    #     Context: The Japanese electronics industry is the lagest in the world.
    #     Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #    Doc: the man went to the store and bought a gallon of milk
    #    Span A: the man went to the
    #    Span B: to the store and bought
    #    Span C: and bought a gallon of
    #    ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature, token_embeddings=None):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["start_positions"] = create_int_feature([feature.start_position])
        features["end_positions"] = create_int_feature([feature.end_position])
        if token_embeddings is not None:
            features["token_embeddings"] = _flat_map_create_float_feature(token_embeddings)

        if self.is_training:
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


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


def _flat_map_create_float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(
        [embedding_value for token_embeddings in values for embedding_value in token_embeddings])))


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

    if FLAGS.write_test:
        writer_fn(TEST_FILE, False, [os.path.join(FLAGS.output_dir, "test.tf_record")], [1.0])

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


def _parse_squad_features(input_file: str,
                          is_training: bool,
                          output_files: [str],
                          splits: [int],
                          max_examples: int = None):

    # STEP 1: Tokenize inputs
    examples = read_squad_examples(input_file=input_file,
                                   is_training=is_training,
                                   max_examples=max_examples)
    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(examples)
    idx = 0
    assert sum(splits) == 1.0
    example_sets = []
    for split in splits:
        next_idx = idx + math.ceil(split * len(examples))
        example_sets.append(examples[idx:next_idx])
        idx = next_idx
    del examples

    for output_file, example_set in zip(output_files, example_sets):
        tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE,
                                               do_lower_case=FLAGS.do_lower_case)

        yield (output_file,
               convert_examples_to_features(examples=example_set,
                                            tokenizer=tokenizer,
                                            max_seq_length=FLAGS.max_seq_length,
                                            doc_stride=FLAGS.doc_stride,
                                            max_query_length=FLAGS.max_query_length,
                                            is_training=is_training))


def write_squad_features(input_file: str,
                         is_training: bool,
                         output_files: [str],
                         splits: [int],
                         max_examples: int = None):
    for output_file, features in _parse_squad_features(input_file, is_training, output_files,
                                                       splits, max_examples):

        writer = FeatureWriter(filename=output_file, is_training=is_training)
        for i, feature in enumerate(features):
            writer.process_feature(feature)

            if i % 1000 == 0:
                print('%d examples processed' % i)

        writer.close()


def write_bert_embeddings(input_file: str,
                          is_training: bool,
                          output_files: [str],
                          splits: [int],
                          max_examples: int = None):
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

    for output_file, features in _parse_squad_features(input_file, is_training, output_files,
                                                       splits, max_examples):
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature

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

        input_fn = bert_input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

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
                print('%d examples processed', ct)

        writer.close()


if __name__ == "__main__":
    tf.app.run()
