import hashlib
import io
import json
import os

import PIL.Image
import contextlib2
import numpy as np
import tensorflow as tf
# use absl for tf 2.0
from absl import app
from absl import flags
from absl import logging
from pycocotools import mask

from . import dataset_util


FLAGS = flags.FLAGS

flags.DEFINE_boolean('include_masks', True,
                     'Whether to include instance segmentations masks (PNG encoded) in the result. default: False.')
flags.DEFINE_string('train_image_dir', 'raw/coco/train2017',
                    'Training image directory.')
flags.DEFINE_string('val_image_dir', 'raw/coco/val2017',
                    'Validation image directory.')
flags.DEFINE_string('test_image_dir', '',
                    'Test image directory.')
flags.DEFINE_string('train_annotations_file', 'raw/coco/annotations/instances_train2017.json',
                    'Training annotations JSON file.')
flags.DEFINE_string('val_annotations_file', 'raw/coco/annotations/instances_val2017.json',
                    'Validation annotations JSON file.')
flags.DEFINE_string('testdev_annotations_file', '',
                    'Test-dev annotations JSON file.')
flags.DEFINE_string('output_dir', 'tfrecord/coco', 'Output data directory.')

logging.set_verbosity(logging.INFO)


def create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, include_masks, num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.
  Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: number of output file shards.
  """

  with contextlib2.ExitStack() as tf_record_close_stack, tf.io.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = dataset_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = dataset_util.create_category_index(
        groundtruth_data['categories'])

    annotations_index = {}
    if 'annotations' in groundtruth_data:
      logging.info(
          'Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    logging.info('%d images are missing annotations.',
                 missing_annotation_count)

    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]

      # Ignore images that only have crowd annotation
      num_crowd = 0
      for object_annotations in annotations_list:
        if object_annotations['iscrowd']:
          num_crowd += 1
      if num_crowd != len(annotations_list):
        _, tf_example, num_annotations_skipped = create_tf_example(image, annotations_list, image_dir,
                                                                   category_index, include_masks)
        total_num_annotations_skipped += num_annotations_skipped
        shard_idx = idx % num_shards
        output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      else:
        logging.info('Image only have crowd annotation ignored')
        total_num_annotations_skipped += len(annotations_list)
    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)


def main(_):
  assert FLAGS.train_image_dir, '`train_image_dir` missing.'
  assert FLAGS.val_image_dir, '`val_image_dir` missing.'
  assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
  assert FLAGS.val_annotations_file, '`val_annotations_file` missing.'

  train_output_path = os.path.join(FLAGS.output_dir, 'coco_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'coco_val.record')

  create_tf_record_from_coco_annotations(
      FLAGS.train_annotations_file,
      FLAGS.train_image_dir,
      train_output_path,
      FLAGS.include_masks,
      num_shards=100)
  create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.val_image_dir,
      val_output_path,
      FLAGS.include_masks,
      num_shards=10)


if __name__ == "__main__":
  app.run(main)
