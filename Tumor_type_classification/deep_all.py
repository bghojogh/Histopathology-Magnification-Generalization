from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
# try:
#     #import special_grads
# except KeyError as e:
#     print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e, file=sys.stderr)

from tensorflow.python.platform import flags
from utils import conv_block, fc, max_pool, lrn, dropout
from utils import xent, kd

FLAGS = flags.FLAGS

class DeepAll:
    def __init__(self, WEIGHTS_PATH, feature_space_dimension):
        """ Call construct_model_*() after initializing deep_all"""
        self.deep_all_lr = FLAGS.deep_all_lr
        self.SKIP_LAYER = ['fc8']
        self.forward = self.forward_alexnet
        self.construct_weights = self.construct_alexnet_weights
        self.loss_func = xent
        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.feature_space_dimension = feature_space_dimension

    def construct_model_train(self, prefix='deep_all_train_'):
        self.inputa = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.inputa1= tf.placeholder(tf.float32)
        self.labela1= tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)
        self.bool_indicator_b_a = tf.placeholder(tf.float32, shape=(FLAGS.num_classes,))
        self.bool_indicator_b_a1 = tf.placeholder(tf.float32, shape=(FLAGS.num_classes,))

        self.KEEP_PROB = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            def cross_entropy(inp, reuse=True):
                # Function to perform meta learning update """
                inputa, inputa1, inputb, labela, labela1, labelb = inp

                # Obtaining the conventional task loss on meta-train
                _, task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                _, task_outputa1 = self.forward(inputa1, weights, reuse=reuse)
                task_lossa1 = self.loss_func(task_outputa1, labela1)
                _, task_outputb = self.forward(inputb, weights, reuse=reuse)
                task_lossb = self.loss_func(task_outputb, labelb)

                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1)) #this accuracy already gathers batch size
                task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), 1), tf.argmax(labela1, 1))
                task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), 1), tf.argmax(labelb, 1))
                task_output = [task_lossa, task_lossa1, task_lossb, task_accuracya, task_accuracya1, task_accuracyb]

                return task_output

            self.global_step = tf.Variable(0, trainable=False)

            input_tensors = (self.inputa, self.inputa1, self.inputb, self.labela, self.labela1, self.labelb)
            result = cross_entropy(inp=input_tensors)
            self.lossa_raw, self.lossa1_raw, self.lossb_raw, accuracya, accuracya1, accuracyb = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
            self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
            self.lossb = avg_lossb = tf.reduce_mean(self.lossb_raw)
            self.source_loss = (avg_lossa + avg_lossa1 + avg_lossb) / 3.0  #--> because we have three training source domains
            self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.deep_all_lr).minimize(self.source_loss, global_step=self.global_step)

            self.accuracya = accuracya * 100.
            self.accuracya1 = accuracya1 * 100.
            self.accuracyb = accuracyb * 100.
            self.source_accuracy = (self.accuracya + self.accuracya1 + self.accuracyb) / 3.0

        ## Summaries
        tf.summary.scalar(prefix+'source_1 loss', self.lossa)
        tf.summary.scalar(prefix+'source_2 loss', self.lossa1)
        tf.summary.scalar(prefix+'source_3 loss', self.lossb)
        tf.summary.scalar(prefix+'source_1 accuracy', self.accuracya)
        tf.summary.scalar(prefix+'source_2 accuracy', self.accuracya1)
        tf.summary.scalar(prefix+'source_3 accuracy', self.accuracyb)

    def construct_model_test(self, prefix='test'):

        self.test_input = tf.placeholder(tf.float32)
        self.test_label = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as testing_scope:
            if 'weights' in dir(self):
                testing_scope.reuse_variables()
                weights = self.weights
            else:
                raise ValueError('Weights not initilized. Create training model before testing model')

            self.embedding_feature, outputs = self.forward(self.test_input, weights)
            losses = self.loss_func(outputs, self.test_label)
            accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(self.test_label, 1))
            self.pred_prob = tf.nn.softmax(outputs)
            self.outputs = outputs

        self.test_loss = losses
        self.test_acc = accuracies

        #--> self.embedding_feature: 4096-dimensional embedding with psi weights (for semantic features)
        #--> self.outputs: FLAGS.num_classes-dimensional embedding with theta weights (after softmax for cross-entropy)
        #--> self.pred_prob: FLAGS.num_classes-dimensional embedding with theta weights (before softmax cross-entropy)
        #--> self.metric_embedding: 256-dimensional embedding with phi weights (for metric features used in triplet loss)

    def load_initial_weights(self, session):
        """Load weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        The weights come as a dict of lists (e.g. weights['conv1'] is a list)
        Load the weights into the model
        """
        weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle= True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope('model', reuse=True):
                    with tf.variable_scope(op_name, reuse=True):

                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=True)
                                session.run(var.assign(data))
                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=True)
                                session.run(var.assign(data))

    def construct_alexnet_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        with tf.variable_scope('conv1') as scope:
            weights['conv1_weights'] = tf.get_variable('weights', shape=[11, 11, 3, 96], initializer=conv_initializer)
            weights['conv1_biases'] = tf.get_variable('biases', [96])

        with tf.variable_scope('conv2') as scope:
            weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 48, 256], initializer=conv_initializer)
            weights['conv2_biases'] = tf.get_variable('biases', [256])

        with tf.variable_scope('conv3') as scope:
            weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 384], initializer=conv_initializer)
            weights['conv3_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv4') as scope:
            weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 384], initializer=conv_initializer)
            weights['conv4_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv5') as scope:
            weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 256], initializer=conv_initializer)
            weights['conv5_biases'] = tf.get_variable('biases', [256])

        # with tf.variable_scope('fc6') as scope:
        #     weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, 4096], initializer=conv_initializer)
        #     weights['fc6_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc6') as scope:
            weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, self.feature_space_dimension], initializer=conv_initializer)
            weights['fc6_biases'] = tf.get_variable('biases', [self.feature_space_dimension])

        # with tf.variable_scope('fc7') as scope:
        #     weights['fc7_weights'] = tf.get_variable('weights', shape=[4096, 4096], initializer=conv_initializer)
        #     weights['fc7_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc7') as scope:
            weights['fc7_weights'] = tf.get_variable('weights', shape=[self.feature_space_dimension, self.feature_space_dimension], initializer=conv_initializer)
            weights['fc7_biases'] = tf.get_variable('biases', [self.feature_space_dimension])

        # with tf.variable_scope('fc8') as scope:
        #     weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, FLAGS.num_classes], initializer=fc_initializer)
        #     weights['fc8_biases'] = tf.get_variable('biases', [FLAGS.num_classes])

        with tf.variable_scope('fc8') as scope:
            weights['fc8_weights'] = tf.get_variable('weights', shape=[self.feature_space_dimension, FLAGS.num_classes], initializer=fc_initializer)
            weights['fc8_biases'] = tf.get_variable('biases', [FLAGS.num_classes])

        return weights

    def forward_alexnet(self, inp, weights, reuse=False):
        # reuse is for the normalization parameters.

        conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'], stride_y=4, stride_x=4, groups=1, reuse=reuse, scope='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75)
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv_block(pool1, weights['conv2_weights'], weights['conv2_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75)
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv_block(pool2, weights['conv3_weights'], weights['conv3_biases'], stride_y=1, stride_x=1, groups=1, reuse=reuse, scope='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv_block(conv3, weights['conv4_weights'], weights['conv4_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv_block(conv4, weights['conv5_weights'], weights['conv5_biases'], stride_y=1, stride_x=1, groups=2, reuse=reuse, scope='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, weights['fc6_weights'], weights['fc6_biases'], activation='relu')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, weights['fc7_weights'], weights['fc7_biases'], activation='relu')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'])

        return fc7, fc8
