import tensorflow as tf
import CYCLEGAN.GANs.operations as ops


class Discriminator(tf.Module):
    def __init__(self, name=None, norm='instance', use_sigmoid=False):
        super(Discriminator, self).__init__(name=name)
        self.norm = norm
        self.use_sigmoid = use_sigmoid
        with self.name_scope:
            self.C64 = ops.c4_k(64, norm, name='C64')
            self.C128 = ops.c4_k(128, norm, name='C128')
            self.C256 = ops.c4_k(256, norm, name='C256')
            self.C512 = ops.c4_k(512, norm, name='C512')
            self.out = ops.last_layer(use_sigmoid=use_sigmoid, name='output')

    @tf.Module.with_name_scope
    def __call__(self, input):
        C64 = self.C64(input)
        C128 = self.C128(C64)
        C256 = self.C256(C128)
        C512 = self.C512(C256)
        output = self.out(C512)
        return output, [C64, C128, C256, C512]
