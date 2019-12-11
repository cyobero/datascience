#!/usr/bin
# Encoder
# The input is put through an embedding which is summed up with the positional encoding.
# The resultingn ooutput of this summation is the input to the encoder layers.
# The output of the encoder is the input to the decoder.
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='encoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units = units,
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout,
            name = 'encoder_layer_{}'.format(i),
        )([outputs, padding_mask])

        return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
