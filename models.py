import tensorflow as tf


# Add docstrings
# CNN tested for CIFAR-10
def cnn_1(input_shape, len_classes=10, activation_1="relu", activation_2="softmax"):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=activation_1, kernel_initializer='he_uniform', 
                                     padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=activation_1, kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(len_classes, activation=activation_2))

    return model


def cnn_2(input_shape, len_classes=10, activation_1="relu", activation_2="softmax"): 
    model = Sequential()
    
    # First Conv layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation=activation_1, padding='same', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4), input_shape=input_shape))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # Second Conv layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation=activation_1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    # Third, fourth, fifth convolution layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation=activation_1, padding='same', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation=activation_1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation=activation_1, padding='same', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # Fully Connected layers
    model.add(Flatten())
    
    model.add(Dense(512, activation=activation_1))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation=activation_1))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation=activation_1))
    model.add(Dropout(0.5))
    
    model.add(Dense(10, activation=activation_2))
    
    model.summary()
    
    return model


# Resnet50
def ResNet50(input_shape, len_classes=10, activation="softmax"):
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)

    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = tf.keras.layers.Dense(len_classes, activation=activation,
                                    name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


# EfficientNetB0
def EfficientNetB0(input_shape, len_classes=10, activation="softmax"):
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)

    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = tf.keras.layers.Dense(len_classes, activation=activation, name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)

    return model
