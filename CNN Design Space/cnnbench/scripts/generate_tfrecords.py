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

"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.

Based on script from
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py

To run:
  python generate_cifar10_tfrecords.py --data_dir=/tmp/cifar-tfrecord
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from six.moves import cPickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds

import shutil

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
        'test': 'validation[:]',
        'resize': 384,
        'crop': 384,
    },
}

def main(dataset, data_dir, tfds_manual_dir):
  print(f'Seeting up dataset: {dataset} into directory: {data_dir}')
  
  data_builder = tfds.builder(dataset, data_dir=data_dir)
  print(f'Dataset info:\n{data_builder.info}')

  if not os.path.exists(os.path.join(data_dir, dataset)):
    os.makedirs(os.path.join(data_dir, dataset))

  ds_info = {}
  for split in data_builder.info.splits.keys():
    ds_info[split] = data_builder.info.splits[split].num_examples
  ds_info['image_shape'] = data_builder.info.features['image'].shape
  ds_info['num_classes'] = data_builder.info.features['label'].num_classes

  print(f'Saved dataset info: {ds_info} to {data_dir}/{dataset}/ds_info.pt')

  ds_file = open(os.path.join(data_dir, f'{dataset}/ds_info.pt'), 'wb')
  pickle.dump(ds_info, ds_file)
  ds_file.close()

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=tfds_manual_dir))

  shutil.rmtree(os.path.join(data_dir, 'downloads'))

  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      type=str,
      default='cifar10',
      help=f'Dataset name, one in: {DATASET_PRESETS.keys()}')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='../../datasets',
      help='Directory to download and extract dataset to')
  parser.add_argument(
      '--tfds_manual_dir',
      type=str,
      default=None,
      help='Directory where dataset is already downloaded (in .tar format)')

  args = parser.parse_args()
  main(args.dataset, args.data_dir, args.tfds_manual_dir)
