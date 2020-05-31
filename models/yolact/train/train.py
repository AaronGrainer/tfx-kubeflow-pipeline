import datetime
import contextlib
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging

import yolact
from data import dataset_coco
from loss import loss_yolact
from utils import learning_rate_schedule


tf.random.set_seed(1234)

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', '../../../data/tfrecord/coco',
                    'directory of tfrecord')
flags.DEFINE_string('weights', './weights',
                    'path to store weights')
flags.DEFINE_integer('train_iter', 800000,
                     'iteraitons')
flags.DEFINE_integer('batch_size', 8,
                     'batch size')
flags.DEFINE_float('lr', 1e-3,
                   'learning rate')
flags.DEFINE_float('momentum', 0.9,
                   'momentum')
flags.DEFINE_float('weight_decay', 5 * 1e-4,
                   'weight_decay')
flags.DEFINE_float('print_interval', 10,
                   'number of iteration between printing loss')
flags.DEFINE_float('save_interval', 10000,
                   'number of iteration between saving model(checkpoint)')
flags.DEFINE_float('valid_iter', 5000,
                   'number of iteration between saving validation weights')


@tf.function
def train_step(model, loss_fn, metrics, optimizer, image, labels):
  with tf.GradientTape() as tape:
    output = model(image, training=True)
    loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(
        output, labels, 91)
  grads = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  metrics.update_state(total_loss)
  return loc_loss, conf_loss, mask_loss, seg_loss


@tf.function
def valid_step(model, loss_fn, metrics, image, labels):
  output = model(image, training=False)
  loc_loss, conf_loss, mask_loss, seg_loss, total_loss = loss_fn(
      output, labels, 91)
  metrics.update_state(total_loss)
  return loc_loss, conf_loss, mask_loss, seg_loss


def main(argv):
  # set up Grappler for graph optimization
  # Ref: https://www.tensorflow.org/guide/graph_optimization
  @contextlib.contextmanager
  def options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
      yield
    finally:
      tf.config.optimizer.set_experimental_options(old_opts)



if __name__ == '__main__':
  app.run(main)
