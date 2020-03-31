import tensorflow as tf
import CYCLEGAN.GANs.operations as ops


class Generator(tf.Module):
    def __init__(self, name=None, num_gen_filters=32, norm='instance', img_size=128):
        super(Generator, self).__init__(name=name)
        self.ngf = num_gen_filters
        with self.name_scope:
            self.c7s1_32 = ops.c7s1_k(self.ngf, norm, act_type='ReLu', name='c7s1_32')
            self.d64 = ops.c3s2_k(self.ngf * 2, norm, name='d64')
            self.d128 = ops.c3s2_k(self.ngf * 4, norm, name='d128')
            self.r0 = ops.res_block(norm, name='r0')
            self.r1 = ops.res_block(norm, name='r1')
            self.r2 = ops.res_block(norm, name='r2')
            self.r3 = ops.res_block(norm, name='r3')
            self.r4 = ops.res_block(norm, name='r4')
            self.r5 = ops.res_block(norm, name='r5')
            self.r6 = ops.res_block(norm, name='r6')
            self.r7 = ops.res_block(norm, name='r7')
            self.r8 = ops.res_block(norm, name='r8')
            self.u64 = ops.upsam_deconv(self.ngf * 2, norm, name='u64')
            self.u32 = ops.upsam_deconv(self.ngf, norm, output_size=img_size, name='u32')
            self.out = ops.c7s1_k(3, norm=None, act_type='tanh', name='output')

    @tf.Module.with_name_scope
    def __call__(self, input):
        c7s1_32 = self.c7s1_32(input)
        d64 = self.d64(c7s1_32)
        d128 = self.d128(d64)
        r0 = self.r0(d128)
        r1 = self.r1(r0)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        r4 = self.r4(r3)
        r5 = self.r5(r4)
        r6 = self.r6(r5)
        r7 = self.r7(r6)
        r8 = self.r8(r7)
        u64 = self.u64(r8)
        u32 = self.u32(u64)
        output = self.out(u32)
        return output

    @tf.function
    def serve_model(self, input):
        generated = self.__call__(input)
        return generated
