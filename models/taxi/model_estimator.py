from absl import logging
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tensorflow_metadata.proto.v0 import schema_pb2
from . import features
from tfx.utils import io_utils


HIDDEN_UNITS = [16, 8]

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40


def _gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _build_estimator(config, hidden_units=None, warm_start_from=None):
  real_valued_columns = [
    tf.feature_column.numeric_column(key, shape=())
    for key in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)
  ]

  categorical_columns = []
  for key in features.transformed_names(features.VOCAB_FEATURE_KEYS):
    categorical_columns.append(
      tf.feature_column.categorical_column_with_identity(
          key,
          num_buckets=features.VOCAB_SIZE + features.OOV_SIZE,
          default_value=0))

  for key, num_buckets in zip(
      features.transformed_names(features.BUCKET_FEATURE_KEYS),
      features.BUCKET_FEATURE_BUCKET_COUNT):
    categorical_columns.append(
      tf.feature_column.categorical_column_with_identity(
        key, num_buckets=num_buckets, default_value=0))

  for key, num_buckets in zip(
      features.transformed_names(features.CATEGORICAL_FEATURE_KEYS),
      features.CATEGORICAL_FEATURE_MAX_VALUES):
    categorical_columns.append(
      tf.feature_column.categorical_column_with_identity(
        key, num_buckets=num_buckets, default_value=0))

  return tf.estimator.DNNLinearCombinedClassifier(
    config=config,
    linear_feature_columns=categorical_columns,
    dnn_feature_columns=real_valued_columns,
    dnn_hidden_units=hidden_units or [100, 70, 50, 25],
    warm_start_from=warm_start_from)


def _example_serving_receiver_fn(tf_transform_output, schema):
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_feature_spec.pop(features.LABEL_KEY)

  raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    raw_feature_spec, default_batch_size=None)
  serving_input_receiver = raw_input_fn()

  transformed_features = tf_transform_output.transform_raw_features(
    serving_input_receiver.features)

  return tf.estimator.export.ServingInputReceiver(
    transformed_features, serving_input_receiver.receiver_tensors)


def _eval_input_receiver_fn(tf_transform_output, schema):
  raw_feature_spec = _get_raw_feature_spec(schema)

  serialized_tf_example = tf.compat.v1.placeholder(
    dtype=tf.string, shape=[None], name='input_example_tensor')

  raw_features = tf.io.parse_example(
    serialized=serialized_tf_example, features=raw_feature_spec)

  transformed_features = tf_transform_output.transform_raw_features(
      raw_features)

  receiver_tensors = {'examples': serialized_tf_example}

  raw_features.update(transformed_features)

  return tfma.export.EvalInputReceiver(
      features=raw_features,
      receiver_tensors=receiver_tensors,
      labels=transformed_features[features.transformed_name(
          features.LABEL_KEY)])


def _input_fn(filenames, tf_transform_output, batch_size=200):
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

  transformed_features = tf.compat.v1.data.make_one_shot_iterator(
      dataset).get_next()
  
  return transformed_features, transformed_features.pop(
      features.transformed_name(features.LABEL_KEY))


def _create_train_and_eval_spec(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  def train_input_fn(): return _input_fn(
    trainer_fn_args.train_files,
    tf_transform_output,
    batch_size=TRAIN_BATCH_SIZE)

  def eval_input_fn(): return _input_fn(
    trainer_fn_args.eval_files,
    tf_transform_output,
    batch_size=EVAL_BATCH_SIZE)

  train_spec = tf.estimator.TrainSpec(
    train_input_fn,
    max_steps=trainer_fn_args.train_steps)

  def serving_receiver_fn(): return _example_serving_receiver_fn(
    tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
    eval_input_fn,
    steps=trainer_fn_args.eval_steps,
    exporters=[exporter],
    name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

  estimator = _build_estimator(
    hidden_units=HIDDEN_UNITS, config=run_config)

  def receiver_fn(): return _eval_input_receiver_fn(
    tf_transform_output, schema)

  return {
    'estimator': estimator,
    'train_spec': train_spec,
    'eval_spec': eval_spec,
    'eval_input_receiver_fn': receiver_fn
  }


def run_fn(fn_args):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  train_and_eval_spec = _create_train_and_eval_spec(fn_args, schema)

  logging.info('Training model.')
  tf.estimator.train_and_evaluate(train_and_eval_spec['estimator'],
                                  train_and_eval_spec['train_spec'],
                                  train_and_eval_spec['eval_spec'])
  logging.info('Training complete.  Model written to %s',
               fn_args.serving_model_dir)

  logging.info('Exporting eval_savedmodel for TFMA.')
  tfma.export.export_eval_savedmodel(
    estimator=train_and_eval_spec['estimator'],
    export_dir_base=fn_args.eval_model_dir,
    eval_input_receiver_fn=train_and_eval_spec['eval_input_receiver_fn'])

  logging.info('Exported eval_savedmodel to %s.', fn_args.eval_model_dir)
