import tensorflow as tf

def model_input():
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, None], name='inputs')
    target_data = tf.placeholder(dtype=tf.float32, shape=[None, None], name='targets')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    return input_data, largets, learning_rate



