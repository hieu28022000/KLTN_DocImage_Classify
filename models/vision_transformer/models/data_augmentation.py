import tensorflow as tf

def create_data_augmentation_layer(image_size):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.Resizing(image_size, image_size),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    return data_augmentation