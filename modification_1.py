from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import time
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', '/home/ubuntu/AiFi/alexnet_summaries', 'Summaries directory')
flags.DEFINE_string('logs_dir', '/home/ubuntu/AiFi/logs', 'Log file directory')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/home/ubuntu/AiFi', 'Data directory')
flags.DEFINE_string('BATCH_SIZE', 100, 'Batch Size')
flags.DEFINE_string('n_epochs', 10, 'Number of epochs')
flags.DEFINE_string('eval_num_steps', 5, 'Number of Steps after which evaluation is done.')


def log(log_file_name, app_name="modification_1"):
    """
    Logger Initialization
    :param log_file_name: Log file name.
    :param app_name: App Name you require.
    :return: Application logger.
    """
    app_logger = logging.getLogger(app_name)
    app_logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler(os.path.join(FLAGS.logs_dir, log_file_name))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    app_logger.addHandler(fh)
    return app_logger

# Creates directory for summarization and logging.
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

if tf.gfile.Exists(FLAGS.logs_dir):
    tf.gfile.DeleteRecursively(FLAGS.logs_dir)
tf.gfile.MakeDirs(FLAGS.logs_dir)

# Logger for logging the results.
logger = log(log_file_name='test_accuracy.log')
# Loads data from tensorflow.
mnist = input_data.read_data_sets(os.path.join(FLAGS.data_dir, 'MNIST_data'), one_hot=True)


def weight_variable(shape):
    """Weights initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Bias Variabble initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def conv2d(x, W):
    """Return a convolutional layer."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """Return max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_conv_layer(input_dim, output_dim, layer_name, input_tensor, act=tf.nn.relu):
    """
    Reusable code for making a convolutional layer.
    It does a matrix multiply, bias add, convolve the input tensor and weights.
    It also applies activation function for non linearity.
    It also sets up name scoping so that the resultant graph is easy to read.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(input_dim)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            logger.info("Biases conv2d = {0}".format(biases.get_shape()))
            logger.info("Preactiavate Conv2d shape = {0}".format(preactivate.get_shape()))
        activations = act(preactivate, 'activation')
        pool_layer = max_pool_2x2(activations)
        return pool_layer


def add_fully_connected_layer(input_dim, output_dim, keep_prob, layer_name, input_tensor, flat=True,
                              dropout=True, act=tf.nn.relu):
    """
    Reusable code for making a fully connected layer.
    It does a matrix multiply, bias add and applies activation function for non-linearity.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            logger.info("Weights of FC1 = {0}".format(weights.get_shape()))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            logger.info("Biases of FC = {0}".format(biases.get_shape()))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            if flat:
                input_tensor_flat = tf.reshape(input_tensor, [-1, input_dim])
                fc = act(tf.matmul(input_tensor_flat, weights) + biases)
                logger.info("Out of FC = {0}".format(fc.get_shape()))
            else:
                fc = act(tf.matmul(input_tensor, weights) + biases)
        # tf.summary.histogram(layer_name + '/activations', fc)
        if dropout:
            fc_drop = tf.nn.dropout(fc, keep_prob)
        else:
            fc_drop = fc
        return fc_drop


def model(x_image, keep_prob):
    """
    Generates the desired Convolutional Neural Network by adding convolutional layers and
    fully connected layers and return the final prediction layer.
    """
    # Conv-1 layer
    h_pool1 = add_conv_layer(input_dim=[3, 3, 1, 64], output_dim=64, layer_name='conv1', input_tensor=x_image)
    logger.info("Shape of Pool1 = {0}".format(h_pool1.get_shape()))
    # Conv-2 Layer
    h_pool2 = add_conv_layer(input_dim=[3, 3, 64, 128], output_dim=128, layer_name='conv2', input_tensor=h_pool1)
    logger.info("Shape of Pool2 = {0}".format(h_pool2.get_shape()))
    # FC-1 layer
    fc_1 = add_fully_connected_layer(input_dim=7 * 7 * 128, output_dim=512, layer_name="fc1",
                                     keep_prob=keep_prob, input_tensor=h_pool2)
    # FC-2 layer
    pred = add_fully_connected_layer(input_dim=512, output_dim=10, keep_prob=keep_prob, layer_name="fc2",
                                     input_tensor=fc_1, flat=False, dropout=False, act=tf.nn.softmax)
    return pred


def train():
    """
    It performs the training of the model and evaluates validation accuracy at specified intervals.
    It also adds a summary of some parameters for visualization in tensorboard.
    """
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    prediction = model(x_image=x_image, keep_prob=keep_prob)
    # Cross entropy function for minimization.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    tf.summary.scalar('cross entropy', cross_entropy)

    # Accuracy Scalar of the model.
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # Summarization for tensorboard.
    merged = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    sess.run(tf.initialize_all_variables())

    n_steps = (mnist.train.num_examples * FLAGS.n_epochs) / (FLAGS.BATCH_SIZE) + 1
    # Actual training
    for i in range(n_steps):
        start_time = time.time()
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.BATCH_SIZE)
        summary, _ = sess.run([merged, train_step], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: FLAGS.dropout})
        train_writer.add_summary(summary, i)
        if i % FLAGS.eval_num_steps == 0:
            summary, acc = sess.run([merged, accuracy],
                                    feed_dict={xs: mnist.validation.images, ys: mnist.validation.labels, keep_prob: 1})
            validation_writer.add_summary(summary, i)
            logger.info('Step: {0} 	Accuracy: {1} 	Time taken: {2}'.format(i, acc, time.time() - start_time))
    test_acc = accuracy.eval(session=sess, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1})
    logger.info('Test Accuracy = {0}'.format(test_acc))
    train_writer.close()
    validation_writer.close()


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
