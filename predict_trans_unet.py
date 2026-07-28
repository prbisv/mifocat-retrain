# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:16:16 2024

@author: ramad
"""

from keras.models import load_model
import cv2 as cv
import os
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import glob
import re

import tensorflow as tf
from tensorflow import math
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import keras
from tensorflow.keras.utils import custom_object_scope

from tensorflow.nn import depth_to_space
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Conv2D, Layer, Dense, Embedding, Dropout, Conv2D, LayerNormalization
from tensorflow.keras.activations import softmax



direktori_citra = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Data Test Resize 128 ED/"
direktori_prediksi_ed = "D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/Hasil Predict Versi 2/transunet/focalctc/"

def get_image_modification_time(item):
    item_path = os.path.join(direktori_citra, item)
    return os.path.getmtime(item_path)


def get_mask_modification_time(item):
    item_path = os.path.join(direktori_mask, item)
    return os.path.getmtime(item_path)

def get_predict_modification_time(item):
    item_path = os.path.join(direktori_prediksi, item)
    return os.path.getmtime(item_path)




class patch_extract(Layer):
    '''
    Extract patches from the input feature map.

    patches = patch_extract(patch_size)(feature_map)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)

    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`

    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

    '''

    def __init__(self, patch_size, **kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):

        batch_size = tf.shape(images)[0]

        patches = extract_patches(images=images,
                                  sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                                  strides=(1, self.patch_size_x, self.patch_size_y, 1),
                                  rates=(1, 1, 1, 1), padding='VALID',)
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)

        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)

        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size,})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class patch_embedding(Layer):
    '''
    Embed patches to tokens.

    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)

    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020.
    An image is worth 16x16 words: Transformers for image recognition at scale.
    arXiv preprint arXiv:2010.11929.

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions.

    Output
    ----------
        embed: Embedded patches.

    For further information see: https://keras.io/api/layers/core_layers/embedding/

    '''

    def __init__(self, num_patch, embed_dim, **kwargs):

        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed

class patch_merging(tf.keras.layers.Layer):
    '''
    Downsample embedded patches; it halfs the number of patches
    and double the embedded dimensions (c.f. pooling layers).

    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions.

    Output
    ----------
        x: downsampled patches.

    '''
    def __init__(self, num_patch, embed_dim, name='', **kwargs):
        super(patch_merging, self).__init__(**kwargs)

        self.num_patch = num_patch
        self.embed_dim = embed_dim

        # A linear transform that doubles the channels
        self.linear_trans = Dense(2*embed_dim, use_bias=False, name='{}_linear_trans'.format(name))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'name':self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):

        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)

        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)

        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))

        # Linear transform
        x = self.linear_trans(x)

        return x

class patch_expanding(tf.keras.layers.Layer):
    '''
    Upsample embedded patches with a given rate (e.g., x2, x4, x8, ...)
    the number of patches is increased, and the embedded dimensions are reduced.

    Input
    ----------
        num_patch: number of patches.
        embed_dim: number of embedded dimensions.
        upsample_rate: the factor of patches expanding,
                       e.g., upsample_rate=2 doubles input patches and halfs embedded dimensions.
        return_vector: the indicator of returning a sequence of tokens (return_vector=True)
                       or two-dimentional, spatially aligned tokens (return_vector=False)

    For further information see: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    '''

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name='patch_expand', **kwargs):
        super(patch_expanding, self).__init__(**kwargs)

        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector

        # Linear transformations that doubles the channels
        self.linear_trans1 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        #
        self.linear_trans2 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        self.prefix = name

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'upsample_rate': self.upsample_rate,
            'return_vector': self.return_vector,
            'name':self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):

        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))

        x = self.linear_trans1(x)

        # rearange depth to number of patches
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))

        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, L*self.upsample_rate*self.upsample_rate, C//2))

        return x

# ========== Swin-Transformer only ========== #
# TODO: documentation.

def window_partition(x, window_size):

    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, H, W, C = x.get_shape().as_list()

    # Subset tensors to patches
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))

    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))

    return windows

def window_reverse(windows, window_size, H, W, C):

    # Reshape a patch sequence to aligned patched
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))

    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))

    return x

def drop_path_(inputs, drop_prob, is_training):

    # Bypass in non-training mode
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    input_shape = tf.shape(inputs)
    batch_num = input_shape[0]; rank = len(input_shape)

    shape = (batch_num,) + (1,) * (rank - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    path_mask = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * path_mask
    return output

class drop_path(Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(drop_path, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({'drop_prob': self.drop_prob})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=None):
        return drop_path_(x, self.drop_prob, training)

class Mlp(tf.keras.layers.Layer):
    def __init__(self, filter_num, drop=0., name='mlp', **kwargs):

        super(Mlp, self).__init__(**kwargs)

        self.filter_num = filter_num
        self.drop = drop

        # MLP layers
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))

        # Dropout layer
        self.drop = Dropout(drop)

        # GELU activation
        self.activation = tf.keras.activations.gelu

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_num': self.filter_num,
            'drop': self.drop,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):

        # MLP --> GELU --> Drop --> MLP --> Drop
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., name='swin_atten', **kwargs):
        super(WindowAttention, self).__init__(**kwargs)

        self.dim = dim # number of input dimensions
        self.window_size = window_size # size of the attention window
        self.num_heads = num_heads # number of self-attention heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # query scaling factor

        self.prefix = name

        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim':self.dim,
            'window_size':self.window_size,
            'num_heads':self.num_heads,
            'qkv_bias':self.qkv_bias,
            'qk_scale':self.qk_scale,
            'attn_drop':self.attn_drop,
            'proj_drop':self.proj_drop,
            'name':self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):

        # zero initialization
        num_window_elements = (2*self.window_size[0] - 1) * (2*self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                            shape=(num_window_elements, self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False, name='{}_attn_pos_ind'.format(self.prefix))

        self.built = True

    def call(self, x, mask=None):

        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C//self.num_heads

        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        # Query rescaling
        q = q * self.scale

        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)

        # Shift window
        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)

        # Dropout after attention
        attn = self.attn_drop(attn)

        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))

        # Linear projection
        x_qkv = self.proj(x_qkv)

        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)

        return x_qkv

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0,
                 num_mlp=1024, qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0,
                 proj_drop=0, drop_path_prob=0, name='swin_block', **kwargs):

        super(SwinTransformerBlock, self).__init__(**kwargs)

        self.dim = dim # number of input dimensions
        self.num_patch = num_patch # number of embedded patches; a tuple of  (heigh, width)
        self.num_heads = num_heads # number of attention heads
        self.window_size = window_size # size of window
        self.shift_size = shift_size # size of window shift
        self.num_mlp = num_mlp # number of MLP nodes
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_prob = drop_path_prob

        self.prefix = name

        # Layers
        self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)

        # Assertions
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'

        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim':self.dim,
            'num_patch':self.num_patch,
            'num_heads':self.num_heads,
            'window_size':self.window_size,
            'shift_size':self.shift_size,
            'num_mlp':self.num_mlp,
            'qkv_bias':self.qkv_bias,
            'qk_scale':self.qk_scale,
            'mlp_drop':self.mlp_drop,
            'attn_drop':self.attn_drop,
            'proj_drop':self.proj_drop,
            'drop_path_prob':self.drop_path_prob,
            'name':self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

            # attention mask
            mask_array = np.zeros((1, H, W, 1))

            ## initialization
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False, name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'

        # Skip connection I (start)
        x_skip = x

        # Layer normalization
        x = self.norm1(x)

        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size * self.window_size, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x

        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H*W, C))

        # Drop-path
        ## if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)

        # Skip connection I (end)
        x = x_skip +  x

        # Skip connection II (start)
        x_skip = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        # Skip connection II (end)
        x = x_skip + x

        return x



def gelu_(X):

    return 0.5*X*(1.0 + math.tanh(0.7978845608028654*(X + 0.044715*math.pow(X, 3))))

def snake_(X, beta):

    return X + (1/beta)*math.square(math.sin(beta*X))


class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU

    Y = GELU()(X)

    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.

    Usage: use it as a tf.keras.Layer


    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape


class Snake(Layer):
    '''
    Snake activation function $X + (1/b)*sin^2(b*X)$. Proposed to learn periodic targets.

    Y = Snake(beta=0.5, trainable=False)(X)

    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions
    and how to fix it. arXiv preprint arXiv:2006.08195.

    '''
    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def membuat_direktori_2d_ed_prediksi(id_pasien, path_direktori):
    
    direktori_citra_1 = os.path.join(path_direktori, "Prediksi Pasien")
    direktori_pasien = os.path.join(direktori_citra_1, f"Pasien {id_pasien}")
    direktori_prediksi = os.path.join(direktori_pasien, "citra prediksi")
    direktori_rv = os.path.join(direktori_pasien, "prediksi right ventricel")
    direktori_myo = os.path.join(direktori_pasien, "prediksi myocardium")
    direktori_lv = os.path.join(direktori_pasien, "prediksi left ventricel")
    
    
    #membuat direktori jika belum ada 
    os.makedirs(direktori_citra_1, exist_ok=True)
    os.makedirs(direktori_pasien, exist_ok=True)
    os.makedirs(direktori_prediksi, exist_ok=True)
    os.makedirs(direktori_rv, exist_ok=True)
    os.makedirs(direktori_myo, exist_ok=True)
    os.makedirs(direktori_lv, exist_ok=True)
    
    return direktori_citra_1, direktori_pasien, direktori_prediksi,  direktori_rv, direktori_myo, direktori_lv


def prediksi_citra(imgs, model):
    
    for img_path in imgs:
        
        #untuk data predict
        match = re.search(r'Pasien(\d+)_', os.path.basename(img_path))
        if match:
            nomor_pasien = match.group(1)
        else:
            print(f"Tidak dapat mengekstrak nomor pasien dari {img_path}")
            continue
        
        direktori_citra_1, direktori_pasien, direktori_prediksi, direktori_rv, direktori_myo, direktori_lv = membuat_direktori_2d_ed_prediksi(nomor_pasien, direktori_prediksi_ed)
        
        img = cv.imread(img_path, 0)
        img_array = np.array(img)
        
        
        train_images = np.repeat(img_array[..., np.newaxis], 3, axis=-1)
        
        # test_images = np.expand_dims(img_array, axis=-1)
        # test_images = test_images / 255.0
        
        # Menyiapkan data untuk prediksi
        # test_img_number = random.randint(0, len(X_test))
        # test_img = X_test[test_img_number]
        # ground_truth = y_test[test_img_number]

        # Mengubah jumlah saluran pada test_img_input agar sesuai dengan model
        test_img_input = np.expand_dims(train_images, axis=0)  # Ubah ini sesuai kebutuhan

        # Melakukan prediksi
        prediction = model.predict(test_img_input)

        # Mengambil label prediksi
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]
        
        
            
        # test_img_norm = test_images[:,:,0][:,:,None]
        # test_img_input = np.expand_dims(test_img_norm, 0)
        # prediction = (model.predict(test_img_input))
        # predicted_img = np.argmax(prediction, axis=3)[0,:,:]
        
        # Normalisasi ke rentang 0 - 255
        predicted_img_norm = (predicted_img / predicted_img.max()) * 255
        predicted_img_uint8 = predicted_img_norm.astype(np.uint8)
        
        
        base_img = os.path.splitext(os.path.basename(img_path))[0]
        
        name_file_predict = os.path.join(direktori_prediksi, str(base_img) + "_predict"+".png")
        name_file_rv = os.path.join(direktori_rv, str(base_img) + "_rv_predict"+".png")
        name_file_myo = os.path.join(direktori_myo, str(base_img) + "_myo_predict"+".png")
        name_file_lv = os.path.join(direktori_lv, str(base_img) + "_lv_predict"+".png")
        
        cv.imwrite(name_file_predict, predicted_img_uint8)
        
        # Menyimpan citra untuk kelas RV
        if 1 in predicted_img:
            predicted_rv = (predicted_img == 1).astype(np.uint8) * 255
            cv.imwrite(name_file_rv, predicted_rv)
        else:
            cv.imwrite(name_file_rv, np.zeros_like(predicted_img_uint8))

        # Menyimpan citra untuk kelas Myo
        if 2 in predicted_img:
            predicted_myo = (predicted_img == 2).astype(np.uint8) * 255
            cv.imwrite(name_file_myo, predicted_myo)
        else:
            cv.imwrite(name_file_myo, np.zeros_like(predicted_img_uint8))

        # Menyimpan citra untuk kelas LV
        if 3 in predicted_img:
            predicted_lv = (predicted_img == 3).astype(np.uint8) * 255
            cv.imwrite(name_file_lv, predicted_lv)
        else:
            cv.imwrite(name_file_lv, np.zeros_like(predicted_img_uint8))
            
            
        
        print(f"Citra {base_img} sudah diproses")
        

    print("Proses Prediksi Selesai....")
        
        
    
def main():
    
    # keras.utils.get_custom_objects()['GELU'] = GELU
    
    # custom_objects = {
    #     'GELU' : GELU
    # }
    
    # with custom_object_scope(custom_objects):
    #     model = load_model('D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Model/model_transunet_cardiac_ed_batch8_categoricalcrossentropy_lr1e-3.hdf5', compile=False)
    
    custom_objects = {
        'patch_extract': patch_extract,
        'GELU': GELU,
        'Snake': Snake,
        'patch_embedding': patch_embedding
    }
    
    # Gunakan custom_objects saat memuat model
    model_path = 'D:/Intelligent Multimedia Network/Research/Riset Bu Dini/Dataset/acdc17/acdc17/Data 2D/ED/Hasil Predict 2D versi 3/model/model transunet epoch 50/model_transunet_cardiac_ed_focalctc_lr1e-3_1.hdf5'
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    
    
    list_img_ed_2d = sorted(glob.glob(direktori_citra + '*/images/Pasien*_*_ed.png'))
    
    prediksi_citra(list_img_ed_2d, model)
    
    
if __name__ == "__main__":
    main()