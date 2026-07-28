import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def conv_block(input_tensor, num_filters):
    """
    Standard encoder/decoder block with 3x3 kernels and ReLU activation[cite: 156, 158].
    """
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet_mifocat(input_shape=(256, 256, 1), num_classes=4):
    """
    U-Net architecture as described in the methodology.
    """
    inputs = layers.Input(input_shape)

    # --- Encoder ---
    # Down-sampling via 2x2 operations 
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bridge
    c5 = conv_block(p4, 1024)

    # --- Decoder ---
    # Symmetric up-sampling to restore full resolution 
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64)

    # Output Layer
    # M=4 classes segmentation map 
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet_MIFOCAT")
    return model

def mifocat_loss(alpha=0.25, gamma=2.0, r1=1.0, r2=1.0, r3=1.0):
    """
    Implementation of the MIFOCAT Unified Loss Function[cite: 5, 127].
    Combines MSE (MI), Focal Loss (FO), and Categorical Cross-Entropy (CAT).
    """
    def loss(y_true, y_pred):
        # 1. Mean Squared Error (MI) [cite: 102]
        # Measures pixel-wise discrepancies
        l_mi = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Categorical Cross-Entropy (CAT) [cite: 121]
        # Standard multi-class classification loss
        l_cat = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        l_cat = tf.reduce_mean(l_cat)

        # 3. Focal Loss (FO) [cite: 109]
        # Addresses class imbalance (alpha=0.25, gamma=2 per [cite: 114])
        # Calculate pixel-wise probabilities
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy term first
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight: alpha * (1 - p_t)^gamma
        weight = alpha * tf.pow((1 - y_pred), gamma)
        
        # Apply weight to cross entropy
        l_fo = tf.reduce_sum(weight * cross_entropy, axis=-1)
        l_fo = tf.reduce_mean(l_fo)

        # Unified Loss Combination 
        # r1, r2, r3 are tunable hyperparameters
        return (r1 * l_mi) + (r2 * l_fo) + (r3 * l_cat)

    return loss

def mean_iou(y_true, y_pred):
    """
    Intersection over Union (IoU) metric implementation.
    """
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())

def dice_score(y_true, y_pred):
    """
    Dice Coefficient metric, complementary to IoU[cite: 91].
    """
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())

# --- Model Compilation ---

# # Define input shape (assuming 256x256 based on 'Image Resizing' step in Fig 1 [cite: 71])
# input_shape = (256, 256, 1) 
# num_classes = 4

# # Initialize model
# model = build_unet_mifocat(input_shape, num_classes)

# # Compile with the Unified MIFOCAT loss and ADAM optimizer 
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss=mifocat_loss(alpha=0.25, gamma=2.0),
#     metrics=['accuracy', mean_iou, dice_score]
# )

# model.summary()

# --- Example Training Call ---
# train_images and train_masks must be prepared (masks one-hot encoded for 4 classes)
# batch_size=8, epochs=50 as per [cite: 167, 168]
# history = model.fit(
#     train_images, 
#     train_masks,
#     batch_size=8,
#     epochs=50,
#     validation_data=(val_images, val_masks)
# )