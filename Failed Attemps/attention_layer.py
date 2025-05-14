import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class SeqWeightedAttention(Layer):
    """
    Implements sequence weighted attention as described in:
    https://arxiv.org/abs/1706.03762
    """
    
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention = return_attention
        super(SeqWeightedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.w = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='uniform',
            trainable=True
        )
        super(SeqWeightedAttention, self).build(input_shape)

    def call(self, x, mask=None):
        # Calculate attention weights
        logits = K.dot(x, self.w)  # Shape: (batch_size, seq_len, 1)
        x_shape = K.shape(x)
        
        # Add mask if provided
        if mask is not None:
            mask = K.cast(mask, K.dtype(logits))
            mask = K.expand_dims(mask, axis=-1)
            logits *= mask
            logits -= (1 - mask) * 1e30

        attention_weights = K.softmax(logits, axis=1)
        
        # Calculate weighted sum
        weighted_sum = K.batch_dot(K.permute_dimensions(x, (0, 2, 1)), attention_weights)
        weighted_sum = K.squeeze(weighted_sum, -1)
        
        if self.return_attention:
            return [weighted_sum, attention_weights]
        return weighted_sum

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(SeqWeightedAttention, self).get_config()
        config.update({'return_attention': self.return_attention})
        return config 