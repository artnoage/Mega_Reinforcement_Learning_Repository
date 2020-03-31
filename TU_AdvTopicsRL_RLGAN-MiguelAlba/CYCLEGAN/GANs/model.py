import tensorflow as tf

import CYCLEGAN.GANs.operations as ops
from CYCLEGAN.GANs.generator import Generator
from CYCLEGAN.GANs.discriminator import Discriminator
from CYCLEGAN.GANs.net_utils import ImagePool

poolA = ImagePool(pool_size=50)
poolB = ImagePool(pool_size=50)


class cycleGAN(tf.Module):
    def __init__(self, config):
        super(cycleGAN, self).__init__()
        self.config = config
        self.real_label = config['LABEL_SMOOTHING']
        self.use_lsgan = config['LS_GAN']
        self.type_loss = config['TYPE_LOSS']
        self.use_sigmoid = not self.use_lsgan
        self.G = Generator('G', num_gen_filters=64, norm=config['TYPE_NORM'], img_size=config['IMG_HEIGHT'])
        self.F = Generator('F', num_gen_filters=64, norm=config['TYPE_NORM'], img_size=config['IMG_HEIGHT'])
        self.Da = Discriminator('Da', norm=config['TYPE_NORM'], use_sigmoid=self.use_sigmoid)
        self.Db = Discriminator('Db', norm=config['TYPE_NORM'], use_sigmoid=self.use_sigmoid)
        self.lamb1 = config['LAMBDA1']
        self.lamb2 = config['LAMBDA2']
        self.schedule_gen = ops.LinearDecay(config['LEARNING_RATE'], config['EPOCHS'] * config['DATA_LENGTH'],
                                            config['START_DECAY'] * config['DATA_LENGTH'])
        self.schedule_dis = ops.LinearDecay(config['LEARNING_RATE'], config['EPOCHS'] * config['DATA_LENGTH'],
                                            config['START_DECAY'] * config['DATA_LENGTH'])
        self.opt1 = tf.optimizers.Adam(learning_rate=self.schedule_gen, beta_1=config['BETA1'])
        self.opt2 = tf.optimizers.Adam(learning_rate=self.schedule_dis, beta_1=config['BETA1'])

    def cycle_consistency_loss(self, G, F, img_a, img_b):
        if self.type_loss == 'SSIM':
            forward_loss = 1 - tf.image.ssim_multiscale(F(G(img_a)), img_a, max_val=2.0)[0]
            backward_loss = 1 - tf.image.ssim_multiscale(G(F(img_b)), img_b, max_val=2.0)[0]
        else:
            forward_loss = tf.reduce_mean(tf.abs(F(G(img_a)) - img_a))
            backward_loss = tf.reduce_mean(tf.abs(G(F(img_b)) - img_b))
        #
        loss = (forward_loss + backward_loss)
        return loss * self.lamb1

    def identity_loss(self, G, F, img_a, img_b):
        id_loss_a = tf.reduce_mean(tf.abs(F(img_a) - img_a))
        id_loss_b = tf.reduce_mean(tf.abs(G(img_b) - img_b))
        loss = (id_loss_a + id_loss_b)
        return loss * self.lamb2

    def discriminator_loss(self, D, img_b, fake_b):
        real_dis, real_feats = D(img_b)
        fake_dis, fake_feats = D(fake_b)
        if self.use_lsgan:
            error_real = tf.reduce_mean(tf.square(real_dis - self.real_label))
            error_fake = tf.reduce_mean(tf.square(fake_dis))
        else:
            error_real = -tf.reduce_mean(ops.safe_log(real_dis))
            error_fake = -tf.reduce_mean(ops.safe_log(1-fake_dis))
        loss = (error_real + error_fake)/2
        return loss

    def generator_loss(self, D, fake_b):
        fake_dis, fake_feats = D(fake_b)
        if self.use_lsgan:
            loss = tf.reduce_mean(tf.square(fake_dis - self.real_label))
        else:
            loss = -tf.reduce_mean(ops.safe_log(fake_dis))/2
        return loss

    @tf.function
    def train_generator(self, img_a, img_b):
        with tf.GradientTape(persistent=True) as tape:
            cycle_loss = self.cycle_consistency_loss(self.G, self.F, img_a, img_b)
            identity_loss = self.identity_loss(self.G, self.F, img_a, img_b)

            fake_b = self.G(img_a)
            g_gan_loss = self.generator_loss(self.Db, fake_b)

            fake_a = self.F(img_b)
            f_gan_loss = self.generator_loss(self.Da, fake_a)

            generator_loss = (g_gan_loss + f_gan_loss) + cycle_loss + identity_loss

        generator_gradients = tape.gradient(generator_loss, self.G.trainable_variables + self.F.trainable_variables)
        assert len(generator_gradients) == len(self.G.trainable_variables + self.F.trainable_variables)
        self.opt1.apply_gradients(zip(generator_gradients, self.G.trainable_variables + self.F.trainable_variables))

        losses = {
            'generator_loss': generator_loss,
            'GENA2B_loss': g_gan_loss,
            'GENB2A_loss': f_gan_loss,
            'cycle_loss': cycle_loss,
            'identity_loss': identity_loss
        }
        return fake_a, fake_b, losses

    @tf.function
    def train_discriminator(self, img_a, img_b, his_a, his_b):
        with tf.GradientTape(persistent=True) as tape:
            d_b_loss = self.discriminator_loss(self.Db, img_b, his_b)
            d_a_loss = self.discriminator_loss(self.Da, img_a, his_a)
            discriminator_loss = (d_a_loss + d_b_loss)

        discriminator_gradients = tape.gradient(discriminator_loss,
                                                self.Da.trainable_variables + self.Db.trainable_variables)
        assert len(discriminator_gradients) == len(self.Da.trainable_variables + self.Db.trainable_variables)
        self.opt2.apply_gradients(zip(discriminator_gradients, self.Da.trainable_variables + self.Db.trainable_variables))

        losses = {
            'discriminator_loss': discriminator_loss,
            'DISA_loss': d_a_loss,
            'DISB_loss': d_b_loss
        }
        return losses

    def train_step(self, img_a, img_b):
        gen_A2B, gen_B2A, opt_generators = self.train_generator(img_a, img_b)

        his_b = poolB(gen_A2B)
        his_a = poolA(gen_B2A)

        opt_discriminators = self.train_discriminator(img_a, img_b, his_a, his_b)
        return opt_generators, opt_discriminators

    def supervise_generations(self, img_a, img_b):
        """
        This function makes available the transformation supervision
        :param img_a: input from domain A
        :param img_b: input from domain B
        :return: list of transformed images
        """
        fake_b = self.G(img_a)
        fake_a = self.F(img_b)
        recons_a = self.F(fake_b)
        recons_b = self.G(fake_a)
        sum_dict = {
            'img_a': img_a,
            'img_b': img_b,
            'fake_a': fake_a,
            'fake_b': fake_b,
            'recons_a': recons_a,
            'recons_b': recons_b
        }
        return sum_dict



