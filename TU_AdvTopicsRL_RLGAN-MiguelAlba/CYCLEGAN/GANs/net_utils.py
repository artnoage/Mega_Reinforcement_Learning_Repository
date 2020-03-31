import os
import glob
import CYCLEGAN.utils as utils

import numpy as np
import tensorflow as tf


def setup_gpus():
    devises = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devises[0], True)


class ImagePool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, images):
        if self.pool_size == 0:
            return images

        out_items = []
        for image in images:
            if len(self.items) < self.pool_size:
                self.items.append(image)
                out_items.append(image)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], image
                    out_items.append(out_item)
                else:
                    out_items.append(image)
        return tf.stack(out_items, axis=0)


class Checkpoint:
    def __init__(self, checkpoint_kwargs, out_dir, max_to_keep=5, keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, out_dir, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)


class data_loader(object):
    def __init__(self, config, name):
        self.config = config
        self.data_dir = config['DATA_DIR']
        self.width = config['IMG_WIDTH']
        self.height = config['IMG_HEIGHT']
        self.depth = config['IMG_DEPTH']
        self.batch_size = config['BATCH_SIZE']
        self.name = name

    def parse_record(self, tf_record):
        feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'depth': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'crop': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        record = tf.io.parse_single_example(tf_record, feature_description)

        crop = tf.io.decode_raw(record['crop'], tf.uint8)
        crop = tf.reshape(crop, [self.height, self.width, 3])
        crop = tf.cast(crop, tf.float32)
        crop = (crop / 127.5) - 1.0
        return crop

    def augment_image(self, crop):
        crop = tf.image.resize(crop, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        crop = tf.image.random_crop(crop, [self.height, self.width, 3])
        crop = tf.image.random_flip_left_right(crop)
        return crop

    def build_dataset(self):
        print('Generating unpaired dataset')
        data_filenames = '{}/{}_*.tfrecord'.format(self.data_dir, self.name)
        num_files = len(glob.glob(data_filenames))
        dataset = tf.data.Dataset.list_files(data_filenames).shuffle(buffer_size=7, seed=15)
        dataset = dataset.interleave(lambda filename: tf.data.TFRecordDataset(filename), cycle_length=num_files)
        dataset = dataset.map(self.parse_record)
        dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=3000 + 3 * self.batch_size, seed=15)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        return dataset


# def merge(images, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[j*h:j*h+h, i*w:i*w+w, :] = image
#     return img


def inverse_transform(images):
    return (images+1.)/2.


def scalars_images(name, data, step):
    if data.shape == ():
        tf.summary.scalar(name, data, step=step)
    if len(data.shape) == 4:
        image = inverse_transform(data.numpy())
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        tf.summary.image(name, image, step=step)
    return None


def tensorboard_output(summary_dict, step, name=None):
    with tf.name_scope(name):
        for name, data in summary_dict.items():
            scalars_images(name, data, step)
    return None


def save_generations(image_dict, out_dir, step):
    images = []
    fn = os.path.join(out_dir, 'train_gen_{}.jpg'.format(step))
    for name, data in image_dict.items():
        new_image = inverse_transform(data.numpy()[0])
        new_image = (new_image * 255.0).astype(np.uint8)
        images.append(new_image)
    merged = np.hstack([img for img in images])
    utils.writeimg(fn, merged)
    return None


if __name__ == '__main__':
    # from train import config
    # from GANs.generator import Generator

    # print(tf.config.experimental.list_physical_devices())
    # print(tf.config.experimental.list_physical_devices())
    # print(tf.test.is_gpu_available())
    # a = tf.Variable(1.0)
    #
    # setup_gpus()
    # G = Generator('G', 32, 'instance', 256)
    # #
    # males = data_loader(config, 'Male_NO')
    # for i, ex in males.build_dataset().enumerate():
    #     if i > 2:
    #         break
    # utils.display_image(G(ex))
    pass
