import os
import numpy as np
import tensorflow as tf


from CYCLEGAN.GANs.net_utils import setup_gpus, inverse_transform


class LoadServedTF(object):
    def __init__(self, level='2'):
        setup_gpus()
        self.level = 'roadfighter-lvl{}'.format(level)
        self.model_path = os.path.join('cyclegan/models/outputs', self.level, 'frozen')
        self.model = tf.saved_model.load(self.model_path)
        self.signature = self.model.signatures['default_serving']

    def transform(self, input):
        input = input[..., None].astype(np.float32)
        input = input / 127.5 - 1
        input = np.array([input])
        input = tf.convert_to_tensor(input)
        out = self.signature(input)['output_0']
        out = inverse_transform(out.numpy()[0])
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

