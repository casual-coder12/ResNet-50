import tensorflow as tf
from tensorflow.keras import layers, models


def res_block(X, filters, s=1, initializer='he_normal', conv=False):
    """
    A residual block for ResNet-50.
    """

    # Unpack the filters
    F1, F2, F3 = filters

    # Get X to use later in residual connection
    X_residual = X
    
    # First convolutional layer
    X = layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    
    # Second convolutional layer
    X = layers.Conv2D(
        filters=F2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    
    # Third convolutional layer
    X = layers.Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=initializer
    )(X)
    X = layers.BatchNormalization()(X)

    # Adjust the dimensions of the residual connection if necessary
    if conv:
        X_residual = layers.Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(s, s),
            padding='valid',
            kernel_initializer=initializer
        )(X_residual)
        X_residual = layers.BatchNormalization()(X_residual)

    # Add the residual connection
    X = layers.Add()([X, X_residual])
    X = layers.Activation('relu')(X)

    return X


def ResNet50(input_shape, num_classes):
    """
    Builds the ResNet-50 architecture.
    """

    # Define the input
    X_input = layers.Input(input_shape)

    # Resize the input to 128x128
    # X = layers.Resizing(128, 128)(X_input)

    if input_shape[0] >= 128:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.05, 0.05),
        ])
    else:
        data_augmentation = tf.keras.Sequential([
            layers.RandomCrop(32, 32),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomTranslation(0.1, 0.1),
        ])

    X = data_augmentation(X_input)

    # Zero-Padding
    X = layers.ZeroPadding2D((3, 3))(X)

    # Stage 1
    img_height = input_shape[0]

    # Case 1: Small images (e.g., MNIST, CIFAR-10)
    if img_height < 64:
        # 3x3 conv (stride 1) without MaxPool
        X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Activation('relu')(X)
    # Case 2: Medium images (exactly 64x64 to 128x128, e.g., Tiny ImageNet)
    elif 64 <= img_height < 128:
        # 3x3 conv (stride 1) + gentle MaxPool for downsampling from 64 to 32
        X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Activation('relu')(X)
        X = layers.MaxPooling2D((2, 2), strides=(2, 2))(X)
    # Case 3: Large images (128x128 and larger, e.g., ImageNet 224x224)
    else:
        # Original ResNet-50 architecture
        X = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
        X = layers.BatchNormalization()(X)
        X = layers.Activation('relu')(X)
        X = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Stage 2
    X = res_block(X, filters=[64, 64, 256], s=1, conv=True)
    X = res_block(X, filters=[64, 64, 256], s=1)
    X = res_block(X, filters=[64, 64, 256], s=1)

    # Stage 3
    X = res_block(X, filters=[128, 128, 512], s=2, conv=True)
    X = res_block(X, filters=[128, 128, 512], s=1)
    X = res_block(X, filters=[128, 128, 512], s=1)
    X = res_block(X, filters=[128, 128, 512], s=1)

    # Stage 4
    X = res_block(X, filters=[256, 256, 1024], s=2, conv=True)
    X = res_block(X, filters=[256, 256, 1024], s=1)
    X = res_block(X, filters=[256, 256, 1024], s=1)
    X = res_block(X, filters=[256, 256, 1024], s=1)
    X = res_block(X, filters=[256, 256, 1024], s=1)
    X = res_block(X, filters=[256, 256, 1024], s=1)

    # Stage 5
    X = res_block(X, filters=[512, 512, 2048], s=2, conv=True)
    X = res_block(X, filters=[512, 512, 2048], s=1)
    X = res_block(X, filters=[512, 512, 2048], s=1)

    # Average Pooling
    # X = layers.AveragePooling2D(pool_size=(2, 2))(X)
    X = layers.MaxPooling2D()(X)

    # Dropout layer to reduce overfitting
    X = layers.Dropout(0.3)(X)

    # Output layer
    X = layers.Flatten()(X)
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create model
    model = models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model