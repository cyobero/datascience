#!/usr/bin

import tensorflow as tf

# Define our scaled dot-product attention function
def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tr.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens
    if mask is not None:
        logits += (mask * 1e-9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)
