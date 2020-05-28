import tensorflow as tf
import tensorflow_transform as tft

from . import features


def _fill_in_missing(x):
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
    tf.sparse.to_dense(
      tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
      default_value),
    axis=1)


def preprocessing_fn(inputs):
  outputs = {}
  for key in features.DENSE_FLOAT_FEATURE_KEYS:
    outputs[features.transformed_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in features.VOCAB_FEATURE_KEYS:
    outputs[features.transformed_name(key)] = tft.compute_and_apply_vocabulary(
        _fill_in_missing(inputs[key]),
        top_k=features.VOCAB_SIZE,
        num_oov_buckets=features.OOV_SIZE)

  for key, num_buckets in zip(features.BUCKET_FEATURE_KEYS,
                              features.BUCKET_FEATURE_BUCKET_COUNT):
    outputs[features.transformed_name(key)] = tft.bucketize(
        _fill_in_missing(inputs[key]),
        num_buckets,
        always_return_num_quantiles=False)

  for key in features.CATEGORICAL_FEATURE_KEYS:
    outputs[features.transformed_name(key)] = _fill_in_missing(inputs[key])

  fare_key = 'fare'
  taxi_fare = _fill_in_missing(inputs[fare_key])
  tips = _fill_in_missing(inputs[features.LABEL_KEY])
  outputs[features.transformed_name(features.LABEL_KEY)] = tf.compat.v1.where(
      tf.math.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      tf.cast(tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
