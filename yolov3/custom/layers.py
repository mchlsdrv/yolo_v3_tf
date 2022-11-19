import tensorflow as tf
import tensorflow_probability as tfp
from keras.engine.base_layer import Layer


class DropBlock(Layer):
    """
    Regularization technique from:

    Ghiasi, Golnaz, Tsung-Yi Lin, and Quoc V. Le. "Dropblock: A regularization method for convolutional networks."
    Advances in neural information processing systems 31 (2018).

    which is extension of the DropOut to Conv layers. It zeros-out blocks of the feature maps, and by doing so helps to releave the
    overfitting problem as the network can rely less on the learned features.
  """

    def __init__(self, keep_prob=.9, block_size=7):
        super().__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        results = inputs
        if kwargs.get('training'):
            _, w, h, c = inputs.shape.as_list()
            gamma = ((1. - self.keep_prob) / self.block_size ** 2) * (w * h / ((w - self.block_size + 1) * (h - self.block_size + 1)))

            sampling_mask_shape = tf.stack([1, h - self.block_size + 1, w - self.block_size + 1, c])
            noise_dist = tfp.distributions.Bernoulli(probs=gamma)
            mask = noise_dist.sample(sampling_mask_shape)

            br = (self.block_size - 1) // 2
            tl = (self.block_size - 1) - br
            pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
            mask = tf.pad(mask, pad_shape)
            mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
            mask = tf.cast(1 - mask, tf.float32)
            results = tf.multiply(inputs, mask)
        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'keep_prob': self.keep_prob,
                'block_size': self.block_size,
                'supports_masking': self.supports_masking,
            }
        )
        return config
