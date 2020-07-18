from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

from . import features


HIDDEN_UNITS = [16, 8]
LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40


def _gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_serve_tf_examples_fn(model, tf_transform_output):
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(features.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    transformed_features.pop(features.transformed_name(features.LABEL_KEY))

    return model(transformed_features)

  return serve_tf_examples_fn


def input_fn(file_pattern, tf_transform_output, batch_size=200):
  transformed_feature_spec = (
    tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=features.transformed_name(features.LABEL_KEY))

  return dataset


def build_keras_model(hidden_units, learning_rate):
  """Creates a DNN Keras model for classifying taxi data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
    learning_rate: [float], learning rate of the Adam optimizer.

  Returns:
    A keras Model.
  """
  real_valued_columns = [
    tf.feature_column.numeric_column(key, shape=())
    for key in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
    tf.feature_column.categorical_column_with_identity(
      key,
      num_buckets=features.VOCAB_SIZE + features.OOV_SIZE,
      default_value=0)
    for key in features.transformed_names(features.VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
    tf.feature_column.categorical_column_with_identity(
      key,
      num_buckets=features.BUCKET_FEATURE_BUCKET_COUNT,
      default_value=0)
    for key in features.transformed_names(features.BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
    tf.feature_column.categorical_column_with_identity(
      key,
      num_buckets=num_buckets,
      default_value=0) 
    for key, num_buckets in zip(
      features.transformed_names(features.CATEGORICAL_FEATURE_KEYS),
      features.CATEGORICAL_FEATURE_MAX_VALUES)
  ]
  indicator_column = [
    tf.feature_column.indicator_column(categorical_column)
    for categorical_column in categorical_columns
  ]

  model = wide_and_deep_classifier(
    wide_columns=indicator_column,
    deep_columns=real_valued_columns,
    dnn_hidden_units=hidden_units,
    learning_rate=learning_rate)
  return model


def wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units,
                              learning_rate):
  input_layers = {
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
    for colname in features.transformed_names(
      features.DENSE_FLOAT_FEATURE_KEYS)
  }
  input_layers.update({
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
    for colname in features.transformed_names(features.VOCAB_FEATURE_KEYS)
  })
  input_layers.update({
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
    for colname in features.transformed_names(features.BUCKET_FEATURE_KEYS)
  })
  input_layers.update({
    colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
    for colname in features.transformed_names(features.CATEGORICAL_FEATURE_KEYS)
  })
  deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
  for numnodes in dnn_hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)
  wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

  output = tf.keras.layers.Dense(
    1, activation='sigmoid')(
      tf.keras.layers.concatenate([deep, wide]))

  model = tf.keras.Model(input_layers, output)
  model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    metrics=[tf.keras.metrics.BinaryAccuracy()])
  model.summary(print_fn=logging.info)
  return model


def run_fn(fn_args):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(fn_args.train_files, tf_transform_output,
                            TRAIN_BATCH_SIZE)
  eval_dataset = input_fn(fn_args.eval_files, tf_transform_output,
                           EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = build_keras_model(
      hidden_units=HIDDEN_UNITS,
      learning_rate=LEARNING_RATE)

  model.fit(train_dataset,
            steps_per_epoch=fn_args.train_steps,
            validation_data=eval_dataset,
            validation_steps=fn_args.eval_steps)

  signatures = {
    'serving_default':
      get_serve_tf_examples_fn(
          model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir,
             save_format='tf', signatures=signatures)
