import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from .map_model import map_model_to_preprocess, map_name_to_handle

class BertModel(tf.keras.models.Model):
    def __init__(self, bert_model_name):
        self.tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        self.tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    def build_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(7, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

# if __name__ == "__main__":
#     bert = BertModel(bert_model_name="small_bert/bert_en_uncased_L-4_H-512_A-8")
#     model = bert.build_model()