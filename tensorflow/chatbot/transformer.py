#!/usr/bin

"""
Transformer consists of the encoder, decoder, and a final linear layer.
The output of the decoder is the input to the linear layer and its output is returned.

enc_padding_mask and dec_padding_mask are usedd to mask out all padding tokens.
look_ahead_mask is used to mask out future tokens in a sequence. As the length of the masks change with
different input sequence lengths, we create these masks with Lambda layers.
"""
def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name='transformer'):
    inputs = tf.keras.Input(shape=(None, ), name='inputs')
    dec_inputs = tf.keras.Input(shape=(None, ), name='dec_inputs')

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size = vocab_size,
        num_layers = num_layers,
        units = units,
        d_model = d_model,
        num_heads = num_heads,
        dropout = dropout,
    )(inputs = [inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size = vocab_size,
        num_layers = num_layers,
        units = units,
        d_model = d_model,
        num_heads = num_heads,
        dropout = dropout,
    ) (inputs = [dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
