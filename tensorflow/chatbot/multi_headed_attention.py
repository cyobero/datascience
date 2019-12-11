#!/usr/bin

# Multihead Attention Layer
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

        def split_heads(self, inputs, batch_size):
            inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(inputs, perm=[0, 2, 1, 3])

        def call(self, inputs):
            query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
            batch_size = tf.reshape(query)[0]

            # linear layers
            query = self.query_dense(query)
            key = self.key_dense(key)
            value = self.value_dense(value)

            # split heads
            query = self.split_heads(query, batch_size)
            key = self.split_heads(key, batch_size)
            value = self.split_heads(value, batch_size)

            scaled_attention = scaled_dot_product_attention(query, key, value, mask)
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

            outputs = self.dense(concat_attention)

            return outputs
