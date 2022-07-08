import tensorflow as tf

from .data_augmentation import create_data_augmentation_layer
from .patches import Patches, PatchEncoder
from .mlp import mlp

def vision_transformer_model(
        input_shape, 
        image_size, 
        patch_size, 
        transformer_layers, 
        num_heads,
        projection_dim,
        mlp_head_units,
        num_classes
    ):
    inputs = tf.keras.layers.Input(shape=input_shape)
    data_augmentation = create_data_augmentation_layer(image_size)
    augmented  = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder((image_size // patch_size) ** 2, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = tf.keras.layers.Dense(num_classes)(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    
    return model