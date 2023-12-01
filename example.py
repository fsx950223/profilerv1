import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import numpy as np

tf.disable_v2_behavior()

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Adding a dimension to the array -> new shape == (28, 28, 1)
# We are doing this because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Getting the images in [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
train_images = np.reshape(train_images, (-1, 784))
test_images = np.reshape(test_images, (-1, 784))

train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')

""" Define placeholder: Where the data will be placed.
Create a two-dimensional tensor for images and correct labels.
None means no limits in length. """
# Placeholder for image data
input_names = []
output_names = []
with tf.xla.experimental.jit_scope():
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [60000, 784], name='x')
        input_names.append(x.name)
        # Placeholder for a correct answer label
        y_input = tf.placeholder(tf.uint8, [60000], name='y_input')
        input_names.append(y_input.name)
        y_ = tf.one_hot(y_input, 10)
        """ Define Variable:  The weight and bias to store learning results """
        # Initialize it to 0.
        W = tf.Variable(tf.zeros([784, 10])) # w is for multiplying a 784 dimensional image vector and yielding a result of 10 dimensions (one hot encoded 0 to 9).
        b = tf.Variable(tf.zeros([10]))      # b is 10 dimensions to be added to the result.

        """ Define model: Softmax regression
        Use Softmax to choose the value with the highest probability among 10 values. """
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        """ Model training """
        # Define Loss function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        # Define a learning rate as 0.5.
        grads = tf.gradients([cross_entropy], [x, y_input])
        # Initialize all variables before starting a session.
        init = tf.global_variables_initializer()
        run_meta = tf.RunMetadata()
        sess.run(init)
        sess.run([cross_entropy], feed_dict={'x:0': train_images, 'y_input:0': train_labels}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)
        output_names = [cross_entropy.name]
        tf.profiler.profile(graph=sess.graph, run_meta=run_meta, options=tf.profiler.ProfileOptionBuilder.time_and_memory())
        graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [cross_entropy.op.name])
        tf.io.gfile.GFile('/tmp/graphdef', 'wb').write(graphdef.SerializeToString())
print(input_names)
print(output_names)