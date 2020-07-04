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

import dataset_util


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


def create_tf_example(image, annotations_list, image_dir,
                      category_index, include_masks=True):
  """Converts image and annotations to a tf.Example proto.
  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  r = 550
  image = image.resize((r, r), PIL.Image.ANTIALIAS)
  bytes_io = io.BytesIO()
  image.save(bytes_io, format='JPEG')
  encoded_jpg = bytes_io.getvalue()
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  num_annotations_skipped = 0

  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_id = int(object_annotations['category_id'])
    category_ids.append(category_id)
    category_names.append(category_index[category_id]['name'].encode('utf8'))
    area.append(object_annotations['area'])

    if include_masks:
      run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                          image_height, image_width)
      binary_mask = mask.decode(run_len_encoding)
      if not object_annotations['iscrowd']:
        binary_mask = np.amax(binary_mask, axis=2)

      pil_image = PIL.Image.fromarray(binary_mask)
      output_io = io.BytesIO()
      pil_image.save(output_io, format='PNG')
      encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
      'image/height': dataset_util.int64_feature(r),
      'image/width': dataset_util.int64_feature(r),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/label_text': dataset_util.bytes_list_feature(category_names),
      'image/object/class/label_id': dataset_util.int64_list_feature(category_ids),
      'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd),
      'image/object/area': dataset_util.float_list_feature(area),
    }

    if include_masks:
      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


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