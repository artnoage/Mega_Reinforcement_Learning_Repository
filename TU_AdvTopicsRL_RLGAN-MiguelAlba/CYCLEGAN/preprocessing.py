import os
import glob
from tqdm import tqdm
import numpy as np

import tensorflow as tf

import CYCLEGAN.utils as utils


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _process_examples(example_data, filename: str, channels=3):
    """
    :param example_data: takes the list of dictionaries and transform them into Tf records, this is an special format
    of tensorflow data that makes your life easier in tf 1.x and 2.0 saving the data and load it in our training loop
    (WARNING: You have to take care of the encoding of features to not have problems when loading the data, this means
    taking into consideration that images are int or float)
    :param filename: output filename
    :param channels: number of channels of the image (RGB=3), grayscale=!
    :return: None
    """
    print(f'Processing {filename} data')
    with tf.io.TFRecordWriter(filename) as writer:
        for index, ex in enumerate(example_data):
            crop = ex['image'].flatten()
            crop = crop.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(ex['image'].shape[0]),
                'width': _int64_feature(ex['image'].shape[1]),
                'depth': _int64_feature(channels),
                'crop': _bytes_feature(crop)
            }))
            writer.write(example.SerializeToString())
    return None


class preprocess_dataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.seed = 15

    def shard_dataset(self, dataset, num_records=500):
        chunk = len(dataset) // num_records
        parts = [(k * chunk) for k in range(len(dataset)) if (k * chunk) < len(dataset)]
        return chunk, parts

    def process_data(self, shard, label, shard_num, num_records):
        data = []
        for idx, fn in enumerate(shard):
            meta = {
                'filename': fn,
                'image': np.load(fn)
            }
            data.append(meta)

        fn = '{}_{:03d}-{:03d}.tfrecord'.format(label, shard_num+1, num_records)
        _process_examples(data,  os.path.join(self.data_path, 'train_data', fn))
        return None

    def save_data(self, dataset_filenames, label):
        # shards
        train_check = 0
        train_chunk, train_parts = self.shard_dataset(dataset_filenames)
        for i, j in enumerate(tqdm(train_parts)):
            train_shard = dataset_filenames[j:(j+train_chunk)]
            self.process_data(train_shard, label, i, 50)
            train_check += len(train_shard)
        print('Number of samples for {} in training: {}'.format(label, train_check))
        return None

    def prep_data(self, level='roadfighter-lvl2'):
        utils.mdir(os.path.join('cyclegan/train_data/{}'.format(level)))
        datasets = glob.glob('{}*'.format(self.data_path))
        for d in datasets:
            label = d.split('-')[1]
            files = glob.glob('{}/*npy'.format(d))
            self.save_data(files, label)
        return None


if __name__ == '__main__':
    prep = preprocess_dataset(data_path='unit/datasets/')
    prep.prep_data()
    pass

