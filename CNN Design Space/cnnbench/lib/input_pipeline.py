# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CIFAR-10 data pipeline with preprocessing.

The data is generated via generate_cifar10_tfrecords.py.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_datasets as tfds

from six.moves import cPickle as pickle
import os

WIDTH = 32
HEIGHT = 32
RGB_MEAN = [125.31, 122.95, 113.87]
RGB_STD = [62.99, 62.09, 66.70]

MAX_IN_MEMORY = 200_000
INCEPTION_CROP = False

DATASET_PRESETS = {
    'cifar10': {
        'train': 'train[:90%]',
        'validation': 'train[90%:]',
        'test': 'test',
        'resize': 384,
        'crop': 384,
    },
    'cifar100': {
        'train': 'train[:90%]',
        'validation': 'train[90%:]',
        'test': 'test',
        'resize': 384,
        'crop': 384,
    },
    'mnist': {
        'train': 'train[:90%]',
        'validation': 'train[90%:]',
        'test': 'test',
        'resize': 384,
        'crop': 384,
    },
    'imagenet2012': {
        'train': 'train[:98%]',
        'validation': 'train[98%:]',
        'test': 'validation',
        'resize': 384,
        'crop': 384,
    },
}


class dataset_input(object):
  """Wrapper class for input_fn passed to Estimator."""

  def __init__(self, mode, config):
    """Initializes a CIFARInput object.

    Args:
      mode: one of [train, valid, test, augment, sample]
      config: config dict built from config.py

    Raises:
      ValueError: invalid mode or data files
    """
    self.preset = DATASET_PRESETS.get(config['dataset'])
    if self.preset is None:
      raise KeyError(f'Please add "{dataset}" to {__name__}.DATASET_PRESETS"')

    self.mode = mode
    self.config = config
    self.repeats = None
    if mode == 'train':         # Training set (no validation & test)
      self.split = self.preset[mode]
    elif mode == 'train_eval':  # For computing train error
      self.split = self.preset['train']
    elif mode == 'valid':       # For computing validation error
      self.split = self.preset['validation']
    elif mode == 'test':        # For computing the test error
      self.split = self.preset['test']
      self.repeats = 1
    elif mode == 'augment':     # Training set (includes validation, no test)
      self.split = f"{self.preset['train']}+{self.preset['validation']}" 
    elif mode == 'sample': 
      self.split = self.preset['validation']
      self.repeats = 1
    else:
      raise ValueError('invalid mode')

    data_builder = tfds.builder(config['dataset'], data_dir = self.config['data_dir'])
    data = data_builder.as_dataset(split = self.split)

    self.num_images = len(list(data))

  def input_fn(self, params):
    """Returns a CIFAR tf.data.Dataset object.

    Args:
      params: parameter dict pass by Estimator.

    Returns:
      tf.data.Dataset object
    """
    batch_size = params['batch_size']
    is_training = (self.mode == 'train' or self.mode == 'augment')
    resize_size = self.preset['resize']
    crop_size = self.preset['crop']
    shuffle_buffer = MAX_IN_MEMORY

    data_builder = tfds.builder(self.config['dataset'], data_dir = self.config['data_dir'])
    data = data_builder.as_dataset(split = self.split)
    
    with open(
      os.path.join(
        self.config['data_dir'], self.config['dataset'], 'ds_info.pt'), 'rb') as ds_file:
      ds_info = pickle.load(ds_file)

    # data = data.map(lambda d: {'image': d['image'], 'label': d['label']})

    # Repeat dataset for training modes
    data = data.repeat(self.repeats)
    if is_training:
      # Shuffle buffer with whole dataset to ensure full randomness per epoch
      data = data.shuffle(min(ds_info['train'], shuffle_buffer))

    def _pp(data):
      im = data['image']
      if is_training:
        if INCEPTION_CROP:
          channels = im.shape[-1]
          begin, size, _ = tf.image.sample_distorted_bounding_box(
              tf.shape(im),
              tf.zeros([0, 0, 4], tf.float32),
              area_range=(0.05, 1.0),
              min_object_covered=0,  # Don't enforce a minimum area.
              use_image_if_no_bounding_boxes=True)
          im = tf.slice(im, begin, size)
          # Unfortunately, the above operation loses the depth-dimension. So we
          # need to restore it the manual way.
          im.set_shape([None, None, channels])
          im = tf.image.resize(im, [crop_size, crop_size])
        else:
          im = tf.image.resize(im, [resize_size, resize_size])
          im = tf.image.random_crop(im, [crop_size, crop_size, 3])
          im = tf.image.flip_left_right(im)
      else:
        # Usage of crop_size here is intentional
        im = tf.image.resize(im, [crop_size, crop_size])
      im = (im - 127.5) / 127.5
      label = tf.one_hot(data['label'], ds_info['num_classes'])  # pylint: disable=no-value-for-parameter
      return {'image': im, 'label': label}

    data = data.map(lambda d: {'image': d['image'], 'label': d['label']})
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)

    data = data.batch(batch_size, drop_remainder=True)

    # # Prefetch to overlap in-feed with training
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


