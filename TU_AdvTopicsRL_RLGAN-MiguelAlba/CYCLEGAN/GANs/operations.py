import tensorflow as tf


def weights(name, shape, mean=0.0, stddev=0.02):
    var = tf.Variable(tf.random_normal_initializer(mean=mean, stddev=stddev)(shape), name=name)
    return var


def bias(name, shape, constant=0.0):
    var = tf.Variable(tf.constant_initializer(constant)(shape), name=name)
    return var


def _leaky_relu(input, slope):
    return tf.maximum(slope*input, input)


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)


def type_norm(norm):
    if norm == 'instance':
        normal = instance_norm()
        return normal
    if norm is None:
        return lambda x: x


class instance_norm(tf.Module):
    def __init__(self, epsilon=1e-5):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    @tf.Module.with_name_scope
    def __call__(self, input):
        conv_shape = input.get_shape()[-1]
        if not hasattr(self, 'scale'):
            self.scale = weights('scale', [conv_shape], mean=1.0)
            self.offset = bias('offset', [conv_shape])

        mean, variance = tf.nn.moments(input, [1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (input - mean) * inv
        return self.scale*normalized + self.offset


class c7s1_k(tf.Module):
    def __init__(self, filters, norm, act_type='ReLu', name=None):
        super(c7s1_k, self).__init__(name=name)
        self.filters = filters
        self.norm = norm
        self.act_type = act_type
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (7, 7, input.get_shape()[-1], self.filters))
        padded = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(padded, self.weights, strides=[1, 1, 1, 1], padding='VALID')

        normalized = self.normalization(conv)

        if self.act_type == 'ReLu':
            output = tf.nn.relu(normalized)
        elif self.act_type == 'tanh':
            output = tf.nn.tanh(normalized)
        return output


class c3s2_k(tf.Module):
    def __init__(self, filters, norm, name=None):
        super(c3s2_k, self).__init__(name=name)
        self.filters = filters
        self.norm = norm
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (3, 3,  input.get_shape()[-1], self.filters))

        conv = tf.nn.conv2d(input, self.weights, strides=[1, 2, 2, 1], padding='SAME')
        normalized = self.normalization(conv)
        output = tf.nn.relu(normalized)
        return output


class paddedConv2D(tf.Module):
    def __init__(self, norm, act_type=None, name=None):
        super(paddedConv2D, self).__init__(name=name)
        self.norm = norm
        self.act_type = act_type
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        self.filters = input.get_shape()[-1]
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (3, 3, input.get_shape()[-1], self.filters))
        padded = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(padded, self.weights, strides=[1, 1, 1, 1], padding='VALID')
        normalized = self.normalization(conv)
        if self.act_type is not None:
            out = tf.nn.relu(normalized)
        else:
            out = normalized
        return out


class res_block(tf.Module):
    def __init__(self, norm, name=None):
        super(res_block, self).__init__(name=name)
        self.norm = norm
        with self.name_scope:
            self.padded1 = paddedConv2D(norm, act_type='ReLu', name='layer1')
            self.padded2 = paddedConv2D(norm, act_type=None, name='layer2')

    @tf.Module.with_name_scope
    def __call__(self, input):
        layer1 = self.padded1(input)
        layer2 = self.padded2(layer1)
        out = input + layer2
        return out


class residule_block(tf.Module):
    def __init__(self, norm, name=None):
        super(residule_block, self).__init__(name=name)
        self.norm = norm
        with self.name_scope:
            self.normalization1 = type_norm(norm)
            self.normalization2 = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        self.filters = input.get_shape()[-1]  # number of filters last conv op
        with tf.name_scope('layer1'):
            if not hasattr(self, 'weights1'):
                self.weights1 = weights('weights1', (3, 3, input.get_shape()[-1], self.filters))
            padded1 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv1 = tf.nn.conv2d(padded1, self.weights1, strides=[1, 1, 1, 1], padding='VALID')
            normalized1 = self.normalization1(conv1)
            relu1 = tf.nn.relu(normalized1)
        with tf.name_scope('layer2'):
            if not hasattr(self, 'weights2'):
                self.weights2 = weights('weights2', (3, 3, relu1.get_shape()[-1], self.filters))
            padded2 = tf.pad(relu1, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            conv2 = tf.nn.conv2d(padded2, self.weights2, strides=[1, 1, 1, 1], padding='VALID')
            normalized2 = self.normalization2(conv2)
            #
        output = input + normalized2
        return output


class dec3s2_k(tf.Module):
    def __init__(self, filters, norm, act_type, output_size=None, name=None):
        super(dec3s2_k, self).__init__(name=name)
        self.filters = filters
        self.norm = norm
        self.act_type = act_type
        self.output_size = output_size
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        input_shape = input.get_shape().as_list()
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (3, 3, self.filters, input_shape[-1]))
        if not self.output_size:
            output_size = input_shape[1]*2
        output_shape = [input_shape[0], output_size, output_size, self.filters]
        deconv = tf.nn.conv2d_transpose(input, self.weights, output_shape=output_shape, strides=[1, 2, 2, 1],
                                        padding='SAME')

        normalized = self.normalization(deconv)
        output = tf.nn.relu(normalized)
        return output


class upsam_deconv(tf.Module):
    def __init__(self, filters, norm, output_size=None, name=None):
        super(upsam_deconv, self).__init__(name=name)
        self.filters = filters
        self.norm = norm
        self.output_size = output_size
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        input_shape = input.get_shape()

        if not self.output_size:
            feat_map_h = input_shape[1] * 2
            feat_map_w = input_shape[2] * 2
        else:
            feat_map_h = self.output_size
            feat_map_w = self.output_size

        up_sampled = tf.image.resize(input, (feat_map_h, feat_map_w), 'nearest')
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (3, 3, up_sampled.get_shape()[-1], self.filters))

        conv = tf.nn.conv2d(up_sampled, self.weights, strides=[1, 1, 1, 1], padding='SAME')
        normalized = self.normalization(conv)
        output = tf.nn.relu(normalized)
        return output


class c4_k(tf.Module):
    def __init__(self, filters, norm, alpha=0.2, stride=[1, 2, 2, 1], name=None):
        super(c4_k, self).__init__(name=name)
        self.filters = filters
        self.norm = norm
        self.alpha = alpha
        self.stride = stride
        with self.name_scope:
            self.normalization = type_norm(norm)

    @tf.Module.with_name_scope
    def __call__(self, input):
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (4, 4, input.get_shape()[-1], self.filters))

        conv = tf.nn.conv2d(input, self.weights, strides=[1, 2, 2, 1], padding='SAME')

        normalized = self.normalization(conv)
        output = tf.nn.leaky_relu(normalized, self.alpha)
        return output


class last_layer(tf.Module):
    def __init__(self, use_sigmoid=False, name=None):
        super(last_layer, self).__init__(name=name)
        self.use_sigmoid = use_sigmoid

    @tf.Module.with_name_scope
    def __call__(self, input):
        if not hasattr(self, 'weights'):
            self.weights = weights('weights', (4, 4, input.get_shape()[-1], 1))
        conv = tf.nn.conv2d(input, self.weights, strides=[1, 1, 1, 1], padding='SAME')

        if self.use_sigmoid:
            output = tf.sigmoid(conv)
        else:
            output = conv
        return output


class LinearDecay(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, max_steps, start_decay):
        super(LinearDecay, self).__init__()
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.start_decay = start_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_lr, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self.start_decay,
            true_fn=lambda: self.initial_lr * (1 - 1 / (self.max_steps - self.start_decay) * (step - self.start_decay)),
            false_fn=lambda: self.initial_lr
        ))
        return self.current_learning_rate


if __name__ == '__main__':
    pass