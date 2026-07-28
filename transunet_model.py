# -*- coding: utf-8 -*-
"""
TransUNet Model for MIFOCAT Cardiac Segmentation
Adapted from predict_trans_unet.py for k-fold cross-validation training

This module provides:
1. All custom layers from TransUNet (patch_extract, patch_embedding, SwinTransformerBlock, etc.)
2. build_transunet_mifocat() function to create the model
3. Custom objects dictionary for model loading

@author: ramad
"""

import numpy as np
import tensorflow as tf
from tensorflow import math
from tensorflow.keras.layers import Layer, Conv2D, Dense, Embedding, Dropout, LayerNormalization
from tensorflow.keras.activations import softmax
from tensorflow.nn import depth_to_space
from tensorflow.image import extract_patches
import tensorflow.keras.backend as K


# ==================== Custom Layers ====================

class patch_extract(Layer):
    '''
    Extract patches from the input feature map.
    patches = patch_extract(patch_size)(feature_map)
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
        patch_dim = patches.shape[-1]
        patch_num = tf.shape(patches)[1]  # Use tf.shape for dynamic dimensions
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))
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
    '''
    def __init__(self, num_patch, embed_dim, name='', **kwargs):
        super(patch_merging, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
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
        
        x = tf.reshape(x, shape=(-1, H, W, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))
        x = self.linear_trans(x)
        return x


class patch_expanding(tf.keras.layers.Layer):
    '''
    Upsample embedded patches with a given rate (e.g., x2, x4, x8, ...)
    '''
    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name='patch_expand', **kwargs):
        super(patch_expanding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.linear_trans1 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
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
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        
        if self.return_vector:
            x = tf.reshape(x, (-1, L*self.upsample_rate*self.upsample_rate, C//2))
        return x


def window_partition(x, window_size):
    _, H, W, C = x.get_shape().as_list()
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W, C):
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, H, W, C))
    return x


def drop_path_(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs
    keep_prob = 1.0 - drop_prob
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
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))
        self.drop = Dropout(drop)
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
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = name
        
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
        num_window_elements = (2*self.window_size[0] - 1) * (2*self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            name='{}_attn_pos'.format(self.prefix),
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(), 
            trainable=True
        )
        
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
        relative_position_index = relative_coords.sum(-1).astype(np.int32)
        
        self.relative_position_index = self.add_weight(
            name='{}_attn_pos_ind'.format(self.prefix),
            shape=relative_position_index.shape,
            dtype=tf.int32,
            initializer=tf.constant_initializer(relative_position_index),
            trainable=False
        )
        
        self.built = True

    def call(self, x, mask=None):
        _, N, C = x.get_shape().as_list()
        head_dim = C//self.num_heads
        
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)
        
        # Shift window
        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_index_flat = tf.cast(relative_position_index_flat, tf.int32)
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)
        
        attn = self.attn_drop(attn)
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.proj_drop(x_qkv)
        
        return x_qkv


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0,
                 num_mlp=1024, qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0,
                 proj_drop=0, drop_path_prob=0, name='swin_block', **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        
        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_prob = drop_path_prob
        self.prefix = name
        
        self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)
        
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'
        
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
            
            mask_array = np.zeros((1, H, W, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)
            
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            attn_mask_np = attn_mask.numpy()
            self.attn_mask = self.add_weight(
                name='{}_attn_mask'.format(self.prefix),
                shape=attn_mask_np.shape,
                initializer=tf.constant_initializer(attn_mask_np),
                trainable=False
            )
        else:
            self.attn_mask = None
        
        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'
        
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, H, W, C))
        
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size * self.window_size, C))
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        
        x = tf.reshape(x, shape=(-1, H*W, C))
        x = self.drop_path(x)
        x = x_skip + x
        
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        
        return x


class GELU(Layer):
    '''Gaussian Error Linear Unit (GELU)'''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return 0.5*inputs*(1.0 + math.tanh(0.7978845608028654*(inputs + 0.044715*math.pow(inputs, 3))))

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape


class Snake(Layer):
    '''Snake activation function'''
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
        return inputs + (1/self.beta_factor)*math.square(math.sin(self.beta_factor*inputs))

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# ==================== Model Building Function ====================

def build_transunet_mifocat(input_shape=(256, 256, 1), num_classes=4, 
                           patch_size=1, embed_dim=512, num_heads=8, 
                           mlp_ratio=4, window_size=4, num_transformer_layers=6,
                           drop_rate=0.1, attn_drop_rate=0.1):
    """
    Build TransUNet model with MIFOCAT loss compatibility.
    
    TransUNet architecture following Chen et al. (2021):
    - CNN Encoder (ResNet-like) for feature extraction
    - Transformer Encoder at bottleneck for global context
    - CNN Decoder with skip connections for segmentation
    
    Reference:
        Chen, J., Lu, Y., Yu, Q., et al. (2021). 
        TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.
        arXiv preprint arXiv:2102.04306.
    
    Args:
        input_shape: Input image shape (height, width, channels). Default (256, 256, 1)
        num_classes: Number of segmentation classes. Default 4 for ACDC dataset
        patch_size: Size of patches for transformer. Default 1 (use feature map directly)
        embed_dim: Transformer embedding dimension. Default 512
        num_heads: Number of attention heads. Default 8
        mlp_ratio: MLP hidden dimension ratio. Default 4
        window_size: Window size for Swin Transformer attention. Default 4
        num_transformer_layers: Number of transformer blocks. Default 6
        drop_rate: Dropout rate. Default 0.1
        attn_drop_rate: Attention dropout rate. Default 0.1
        
    Returns:
        Keras Model with TransUNet architecture
    """
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(input_shape, name='input')
    
    # ========== CNN Encoder (ResNet-like backbone) ==========
    # Stage 1: 256x256 -> 128x128
    c1 = layers.Conv2D(64, (3, 3), padding='same', name='enc1_conv1')(inputs)
    c1 = layers.BatchNormalization(name='enc1_bn1')(c1)
    c1 = layers.Activation('relu', name='enc1_relu1')(c1)
    c1 = layers.Conv2D(64, (3, 3), padding='same', name='enc1_conv2')(c1)
    c1 = layers.BatchNormalization(name='enc1_bn2')(c1)
    c1 = layers.Activation('relu', name='enc1_relu2')(c1)
    p1 = layers.MaxPooling2D((2, 2), name='enc1_pool')(c1)  # 128x128
    
    # Stage 2: 128x128 -> 64x64
    c2 = layers.Conv2D(128, (3, 3), padding='same', name='enc2_conv1')(p1)
    c2 = layers.BatchNormalization(name='enc2_bn1')(c2)
    c2 = layers.Activation('relu', name='enc2_relu1')(c2)
    c2 = layers.Conv2D(128, (3, 3), padding='same', name='enc2_conv2')(c2)
    c2 = layers.BatchNormalization(name='enc2_bn2')(c2)
    c2 = layers.Activation('relu', name='enc2_relu2')(c2)
    p2 = layers.MaxPooling2D((2, 2), name='enc2_pool')(c2)  # 64x64
    
    # Stage 3: 64x64 -> 32x32
    c3 = layers.Conv2D(256, (3, 3), padding='same', name='enc3_conv1')(p2)
    c3 = layers.BatchNormalization(name='enc3_bn1')(c3)
    c3 = layers.Activation('relu', name='enc3_relu1')(c3)
    c3 = layers.Conv2D(256, (3, 3), padding='same', name='enc3_conv2')(c3)
    c3 = layers.BatchNormalization(name='enc3_bn2')(c3)
    c3 = layers.Activation('relu', name='enc3_relu2')(c3)
    p3 = layers.MaxPooling2D((2, 2), name='enc3_pool')(c3)  # 32x32
    
    # Stage 4 (bottleneck): 32x32 -> 16x16
    c4 = layers.Conv2D(embed_dim, (3, 3), padding='same', name='enc4_conv1')(p3)
    c4 = layers.BatchNormalization(name='enc4_bn1')(c4)
    c4 = layers.Activation('relu', name='enc4_relu1')(c4)
    c4 = layers.Conv2D(embed_dim, (3, 3), padding='same', name='enc4_conv2')(c4)
    c4 = layers.BatchNormalization(name='enc4_bn2')(c4)
    c4 = layers.Activation('relu', name='enc4_relu2')(c4)
    p4 = layers.MaxPooling2D((2, 2), name='enc4_pool')(c4)  # 16x16
    
    # ========== Transformer Encoder (at bottleneck) ==========
    # Shape at bottleneck: (batch, 16, 16, 512) for input (256, 256, 1)
    bottleneck_h = input_shape[0] // 16
    bottleneck_w = input_shape[1] // 16
    
    # Extract patches from feature map
    # patch_size=1 means we treat each spatial location as a patch
    patches = patch_extract(patch_size=(patch_size, patch_size), name='patch_extract')(p4)
    # Shape: (batch, num_patches, patch_dim) where num_patches = (16/patch_size)^2
    
    num_patches = (bottleneck_h // patch_size) * (bottleneck_w // patch_size)
    
    # Patch embedding with positional encoding
    patch_emb = patch_embedding(num_patches, embed_dim, name='patch_embedding')(patches)
    # Shape: (batch, num_patches, embed_dim)
    
    # Stack of Swin Transformer Blocks
    x_trans = patch_emb
    for i in range(num_transformer_layers):
        # Alternate between non-shifted and shifted windows
        shift_size = 0 if (i % 2 == 0) else window_size // 2
        x_trans = SwinTransformerBlock(
            dim=embed_dim,
            num_patch=(bottleneck_h // patch_size, bottleneck_w // patch_size),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=int(embed_dim * mlp_ratio),
            qkv_bias=True,
            qk_scale=None,
            mlp_drop=drop_rate,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_prob=drop_rate * (i / num_transformer_layers),
            name=f'swin_block_{i}'
        )(x_trans)
    
    # Reshape back to spatial dimensions
    # (batch, num_patches, embed_dim) -> (batch, H, W, embed_dim)
    H_trans = bottleneck_h // patch_size
    W_trans = bottleneck_w // patch_size
    x_trans_reshaped = layers.Reshape((H_trans, W_trans, embed_dim), name='trans_reshape')(x_trans)
    
    # Project back to original bottleneck size if needed
    if patch_size > 1:
        # Upsample to original bottleneck spatial size
        x_trans_reshaped = layers.UpSampling2D(size=(patch_size, patch_size), 
                                                interpolation='bilinear',
                                                name='trans_upsample')(x_trans_reshaped)
    
    # ========== CNN Decoder (with skip connections from encoder) ==========
    # Decoder Stage 1: 16x16 -> 32x32
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='dec5_upconv')(x_trans_reshaped)
    u5 = layers.concatenate([u5, c4], name='dec5_concat')  # Skip connection from enc4
    c5 = layers.Conv2D(256, (3, 3), padding='same', name='dec5_conv1')(u5)
    c5 = layers.BatchNormalization(name='dec5_bn1')(c5)
    c5 = layers.Activation('relu', name='dec5_relu1')(c5)
    c5 = layers.Conv2D(256, (3, 3), padding='same', name='dec5_conv2')(c5)
    c5 = layers.BatchNormalization(name='dec5_bn2')(c5)
    c5 = layers.Activation('relu', name='dec5_relu2')(c5)
    
    # Decoder Stage 2: 32x32 -> 64x64
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='dec6_upconv')(c5)
    u6 = layers.concatenate([u6, c3], name='dec6_concat')  # Skip connection from enc3
    c6 = layers.Conv2D(256, (3, 3), padding='same', name='dec6_conv1')(u6)
    c6 = layers.BatchNormalization(name='dec6_bn1')(c6)
    c6 = layers.Activation('relu', name='dec6_relu1')(c6)
    c6 = layers.Conv2D(256, (3, 3), padding='same', name='dec6_conv2')(c6)
    c6 = layers.BatchNormalization(name='dec6_bn2')(c6)
    c6 = layers.Activation('relu', name='dec6_relu2')(c6)
    
    # Decoder Stage 3: 64x64 -> 128x128
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='dec7_upconv')(c6)
    u7 = layers.concatenate([u7, c2], name='dec7_concat')  # Skip connection from enc2
    c7 = layers.Conv2D(128, (3, 3), padding='same', name='dec7_conv1')(u7)
    c7 = layers.BatchNormalization(name='dec7_bn1')(c7)
    c7 = layers.Activation('relu', name='dec7_relu1')(c7)
    c7 = layers.Conv2D(128, (3, 3), padding='same', name='dec7_conv2')(c7)
    c7 = layers.BatchNormalization(name='dec7_bn2')(c7)
    c7 = layers.Activation('relu', name='dec7_relu2')(c7)
    
    # Decoder Stage 4: 128x128 -> 256x256
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='dec8_upconv')(c7)
    u8 = layers.concatenate([u8, c1], name='dec8_concat')  # Skip connection from enc1
    c8 = layers.Conv2D(64, (3, 3), padding='same', name='dec8_conv1')(u8)
    c8 = layers.BatchNormalization(name='dec8_bn1')(c8)
    c8 = layers.Activation('relu', name='dec8_relu1')(c8)
    c8 = layers.Conv2D(64, (3, 3), padding='same', name='dec8_conv2')(c8)
    c8 = layers.BatchNormalization(name='dec8_bn2')(c8)
    c8 = layers.Activation('relu', name='dec8_relu2')(c8)
    
    # ========== Output Layer ==========
    # Final 1x1 convolution for class predictions
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', name='output')(c8)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name="TransUNet_MIFOCAT")
    return model


# ==================== Custom Objects for Model Loading ====================

def get_custom_objects():
    """
    Returns dictionary of custom objects for loading TransUNet models.
    Use this when loading saved models with keras.models.load_model()
    
    Example:
        from transunet_model import get_custom_objects
        model = load_model('model.h5', custom_objects=get_custom_objects())
    """
    return {
        'patch_extract': patch_extract,
        'patch_embedding': patch_embedding,
        'patch_merging': patch_merging,
        'patch_expanding': patch_expanding,
        'drop_path': drop_path,
        'Mlp': Mlp,
        'WindowAttention': WindowAttention,
        'SwinTransformerBlock': SwinTransformerBlock,
        'GELU': GELU,
        'Snake': Snake,
    }
