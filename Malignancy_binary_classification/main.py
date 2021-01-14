import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import ImageDataGenerator
from masf_func import MASF
from deep_all import DeepAll
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

if not str('0'):
    print('No GPU given... setting to 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json

with open('local_roots.json') as json_file:
    local_roots = json.load(json_file)

##-------------------- settings and paths:

## Dataset settings:
flags.DEFINE_string('dataset', 'pathology_binary', 'set the dataset')  # --> pacs, pathology, pathology_binary
if FLAGS.dataset == "pacs":
    flags.DEFINE_string('target_domain', 'art_painting', 'set the target domain')
    flags.DEFINE_integer('num_classes', 7, 'number of classes used in classification.')
    class_list = {'0': 'dog',
                  '1': 'elephant',
                  '2': 'giraffe',
                  '3': 'guitar',
                  '4': 'horse',
                  '5': 'house',
                  '6': 'person'}
    domain_list = ['art_painting', 'cartoon', 'photo', 'sketch']
elif FLAGS.dataset == "pathology":
    flags.DEFINE_string('target_domain', '200', 'set the target domain')
    flags.DEFINE_integer('num_classes', 8, 'number of classes used in classification.')
    class_list = {'0': 'benign adenosis',
                  '1': 'benign fibroadenoma',
                  '2': 'benign phyllodes tumor',
                  '3': 'benign tubular adenoma',
                  '4': 'malignant ductal carcinoma',
                  '5': 'malignant lobular carcinoma',
                  '6': 'malignant mucinous carcinoma',
                  '7': 'malignant papillary carcinoma'}
    domain_list = ['40', '100', '200', '400']
elif FLAGS.dataset == "pathology_binary":
    flags.DEFINE_string('target_domain', '400', 'set the target domain')
    flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification.')
    class_list = {'0': 'benign',
                  '1': 'malignant'}
    domain_list = ['40', '100', '200', '400']
## computer-dependent paths:
if FLAGS.dataset == "pacs":
    flags.DEFINE_string('dataroot', local_roots['dataroot_pacs'],
                        'Root folder where PACS dataset (Raw images) is stored')
    flags.DEFINE_string('filelist_root', local_roots['filelist_root_pacs'],
                        'path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line')
elif FLAGS.dataset == "pathology":
    flags.DEFINE_string('dataroot', local_roots['dataroot_pathology'],
                        'Root folder where PACS dataset (Raw images) is stored')
    flags.DEFINE_string('filelist_root', local_roots['filelist_root_pathology'],
                        'path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line')
elif FLAGS.dataset == "pathology_binary":
    flags.DEFINE_string('dataroot', local_roots['dataroot_pathology'],
                        'Root folder where PACS dataset (Raw images) is stored')
    flags.DEFINE_string('filelist_root', local_roots['filelist_root_pathology_binary'],
                        'path to .txt files (e.g., art_painting.txt, cartoon.txt) where images are listed line by line')
flags.DEFINE_string('WEIGHTS_PATH', local_roots['WEIGHTS_PATH'], 'path for the AlexNet pre-trained weights.')
## material of json file in Milad's and Benyamin's computers, respectively:
# {"dataroot": "D:/Projects/masf/PACS/kfold/", "WEIGHTS_PATH": "D:/Projects/masf/pretrained_weights/bvlc_alexnet.npy", "filelist_root": "D:/Projects/masf/txt_files/splits/"}
# {"dataroot_pacs": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/PACS/PACS_DataSet/kfold/", "WEIGHTS_PATH": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/AlexNet_weights/bvlc_alexnet.npy", "filelist_root_pacs": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/PACS/PACS_DataSet/txt_files/splits/",
#  "dataroot_pathology": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/Pathology_dataset/normalized_dataset_cropped/kfold/", "filelist_root_pathology": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/Pathology_dataset/normalized_dataset_cropped/txt_files/splits/"
#  "filelist_root_pathology_binary": "C:/Users/benya/Desktop/my_PhD/MAML/datasets/Pathology_dataset/normalized_dataset_cropped/txt_files/splits_binary/"}

## Training options
flags.DEFINE_integer('train_iterations', 4000, 'number of training iterations.')
flags.DEFINE_integer('meta_batch_size', 24, 'number of images sampled per source domain (in MASF)')
flags.DEFINE_integer('deep_all_batch_size', 24, 'number of images sampled per source domain (in DeepAll)')
flags.DEFINE_float('inner_lr', 1e-5, 'step size alpha for inner gradient update on meta-train')
flags.DEFINE_float('outer_lr', 1e-5, 'learning rate for outer updates with (task-loss + meta-loss)')
flags.DEFINE_float('metric_lr', 1e-5, 'learning rate for the metric embedding nn with AdamOptimizer')
flags.DEFINE_float('deep_all_lr', 1e-5, 'learning rate for the deep_all network with AdamOptimizer')
flags.DEFINE_float('margin', 10, 'distance margin in metric loss')
flags.DEFINE_bool('clipNorm', True, 'if True, gradients clip by Norm, otherwise, gradients clip by value')
flags.DEFINE_float('gradients_clip_value', 2.0, 'clip_by_value for SGD computing new theta at meta loss')
flags.DEFINE_float('feature_space_dimension', 4096,
                   'feature space dimensionality in neural network (one-to-last layer)')
flags.DEFINE_bool('masf_mode', True, 'if True, it trains MASF; otherwise, it trains the DeepAll')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './log/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_integer('which_checkpoint_to_load', 2300,
                     'the checkpoint to be loaded for resuming training or the test phase')
flags.DEFINE_integer('summary_interval', 50, 'frequency for logging training summaries')
flags.DEFINE_integer('save_interval', 50, 'intervals to save model')
flags.DEFINE_integer('print_interval', 50, 'intervals to print out training info')
flags.DEFINE_integer('test_print_interval', 50, 'intervals to test the model')
flags.DEFINE_integer('val_print_interval', 50, 'intervals to test the model')


##-------------------- main:

def main():
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    if FLAGS.dataset == "pacs":
        domain_list = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif FLAGS.dataset == "pathology" or FLAGS.dataset == "pathology_binary":
        domain_list = ['40', '100', '200', '400']
    all_domains_list = domain_list.copy()
    domain_list.remove(FLAGS.target_domain)
    source_domains_list = domain_list

    if FLAGS.masf_mode:
        exp_string = 'MASF_' + FLAGS.target_domain + '.mbs_' + str(FLAGS.meta_batch_size) + \
                     '.inner' + str(FLAGS.inner_lr) + '.outer' + str(FLAGS.outer_lr) + '.clipNorm' + str(
            FLAGS.gradients_clip_value) + \
                     '.metric' + str(FLAGS.metric_lr) + '.margin' + str(FLAGS.margin)
    else:
        exp_string = 'DeepAll_' + FLAGS.target_domain + '.mbs_' + str(FLAGS.deep_all_batch_size) + \
                     '.lr' + str(FLAGS.deep_all_lr) + '.margin' + str(FLAGS.margin)

    # Constructing model
    FLAGS.feature_space_dimension = int(FLAGS.feature_space_dimension)
    if FLAGS.masf_mode:
        model = MASF(FLAGS.WEIGHTS_PATH, FLAGS.feature_space_dimension)
    else:
        model = DeepAll(FLAGS.WEIGHTS_PATH, FLAGS.feature_space_dimension)
    model.construct_model_train()
    model.construct_model_test()

    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                           max_to_keep=None)  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print('Loading pretrained weights')
    model.load_initial_weights(sess)

    resume_itr = 0
    if FLAGS.resume or not FLAGS.train:
        checkpoint_dir = FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + '/' + str(
            FLAGS.which_checkpoint_to_load) + '/'
        load_network_model(saver_=saver, session_=sess, checkpoint_dir=checkpoint_dir)
        resume_itr = FLAGS.which_checkpoint_to_load + 1

    train_file_list = [os.path.join(FLAGS.filelist_root, source_domain + '_train_kfold.txt') for source_domain in
                       source_domains_list]
    test_file_list = [os.path.join(FLAGS.filelist_root, FLAGS.target_domain + '_test_kfold.txt')]
    val_file_list = [os.path.join(FLAGS.filelist_root, FLAGS.target_domain + '_crossval_kfold.txt')]
    evaluation_file_list = [os.path.join(FLAGS.filelist_root, domain_ + '_test_kfold.txt') for domain_ in
                            all_domains_list]

    if FLAGS.train:
        if FLAGS.masf_mode:
            train_MASF(model, saver, sess, exp_string, train_file_list, test_file_list[0], val_file_list[0], resume_itr)
        else:
            train_DeepAll(model, saver, sess, exp_string, train_file_list, test_file_list[0], val_file_list[0],
                          resume_itr)
    else:
        evaluation_after_training(model, sess, evaluation_file_list, evaluation_batch_size=1)


##-------------------- functions:

def train_MASF(model, saver, sess, exp_string, train_file_list, test_file, val_file, resume_itr=0):
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string, sess.graph)
    source_losses, target_losses, source_accuracies, target_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [], [], []
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], dataroot=FLAGS.dataroot, mode='training', \
                                         batch_size=FLAGS.meta_batch_size, num_classes=FLAGS.num_classes, shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(
                tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

        test_data = ImageDataGenerator(test_file, dataroot=FLAGS.dataroot, mode='inference', \
                                       batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()

        val_data = ImageDataGenerator(val_file, dataroot=FLAGS.dataroot, mode='inference', \
                                      batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        val_iterator = tf.data.Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)
        val_next_batch = val_iterator.get_next()

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size / FLAGS.meta_batch_size)))

    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))

    val_init_op = val_iterator.make_initializer(val_data.data)
    val_batches_per_epoch = int(np.floor(val_data.data_size / 1))

    # Training begins
    best_test_acc, best_val_acc = 0, 0
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sampling training and test tasks
        num_training_tasks = len(train_file_list)
        num_meta_train = num_training_tasks - 1
        num_meta_test = num_training_tasks - num_meta_train  # as setting num_meta_test = 1

        # Randomly choosing meta train and meta test domains
        task_list = np.random.permutation(num_training_tasks)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[num_meta_train:]

        for i in range(len(train_file_list)):
            if itr % train_batches_per_epoch[i] == 0 or resume_itr != 0:
                sess.run(training_init_op[i])  # initialize training sample generator at itr=0

        # Sampling meta-train, meta-test data
        for i in range(num_meta_train):
            task_ind = meta_train_index_list[i]
            if i == 0:
                inputa, labela = sess.run(train_next_list[task_ind])
            elif i == 1:
                inputa1, labela1 = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-train domains.')

        for i in range(num_meta_test):
            task_ind = meta_test_index_list[i]
            if i == 0:
                inputb, labelb = sess.run(train_next_list[task_ind])
            else:
                raise RuntimeError('check number of meta-test domains.')

        # to avoid a certain un-sampled class affect stability of global class alignment
        # i.e., mask-out the un-sampled class from computing kd-loss
        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda = np.unique(np.argmax(labela, axis=1))
        bool_indicator_b_a = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            # only count class that are sampled in both source domains
            if (i in sampledb) and (i in sampleda):
                bool_indicator_b_a[i] = 1.0

        sampledb = np.unique(np.argmax(labelb, axis=1))
        sampleda1 = np.unique(np.argmax(labela1, axis=1))
        bool_indicator_b_a1 = [0.0] * FLAGS.num_classes
        for i in range(FLAGS.num_classes):
            if (i in sampledb) and (i in sampleda1):
                bool_indicator_b_a1[i] = 1.0

        part = FLAGS.meta_batch_size / 3
        part = int(part)
        input_group = np.concatenate((inputa[:part], inputa1[:part], inputb[:part]), axis=0)
        label_group = np.concatenate((labela[:part], labela1[:part], labelb[:part]), axis=0)
        group_list = np.sum(label_group, axis=0)
        label_group = np.argmax(label_group, axis=1)  # transform one-hot labels into class-wise integer

        feed_dict = {model.inputa: inputa, model.labela: labela, \
                     model.inputa1: inputa1, model.labela1: labela1, \
                     model.inputb: inputb, model.labelb: labelb, \
                     model.input_group: input_group, model.label_group: label_group, \
                     model.bool_indicator_b_a: bool_indicator_b_a, model.bool_indicator_b_a1: bool_indicator_b_a1,
                     model.KEEP_PROB: 0.5}

        output_tensors = [model.task_train_op, model.meta_train_op, model.metric_train_op]
        output_tensors.extend(
            [model.summ_op, model.global_loss, model.source_loss, model.source_accuracy, model.metric_loss])
        _, _, _, summ_writer, global_loss, source_loss, source_accuracy, metric_loss = sess.run(output_tensors,
                                                                                                feed_dict)

        source_losses.append(source_loss)
        source_accuracies.append(source_accuracy)

        if itr % FLAGS.print_interval == 0:
            print('---' * 10 + '\n%s' % exp_string)
            print('number of samples per category:', group_list)
            print('global loss: %.7f' % global_loss)
            print('metric loss: %.7f ' % metric_loss)
            print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
            print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
            # log loss and accuracy:
            path_save_train_acc = os.path.join(FLAGS.logdir, FLAGS.dataset, exp_string)
            if not os.path.exists(path_save_train_acc):
                os.makedirs(path_save_train_acc)
            with open(path_save_train_acc + '/eva_' + 'train' + '.txt', 'a') as fle:
                fle.write(
                    'Train results: Iteration %d, global loss: %.7f, metric loss: %.7f, Loss: %f, Accuracy: %f \n' % (
                    itr, global_loss, metric_loss, np.mean(source_losses), np.mean(source_accuracies)))
            source_losses, target_losses = [], []

        if itr % FLAGS.summary_interval == 0 and FLAGS.log:
            train_writer.add_summary(summ_writer, itr)

        if itr % FLAGS.save_interval == 0:
            # saver.save(sess, FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + '/model' + str(itr))
            checkpoint_dir = FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + "/" + str(itr) + "/"
            save_network_model(saver_=saver, session_=sess, checkpoint_dir=checkpoint_dir,
                               model_name="model_itr" + str(itr))

        # Testing periodically:
        if itr % FLAGS.val_print_interval == 0:
            val_acc, best_val_acc = evaluation_during_training(sess, model, exp_string, val_batches_per_epoch,
                                                               val_init_op, val_next_batch, itr, best_val_acc, saver,
                                                               is_val=True)
        if itr % FLAGS.test_print_interval == 0:
            test_acc, best_test_acc = evaluation_during_training(sess, model, exp_string, test_batches_per_epoch,
                                                                 test_init_op, test_next_batch, itr, best_test_acc,
                                                                 saver, is_val=False)


def train_DeepAll(model, saver, sess, exp_string, train_file_list, test_file, val_file, resume_itr=0):
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string, sess.graph)
    source_losses, target_losses, source_accuracies, target_accuracies = [], [], [], []

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [], [], []
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], dataroot=FLAGS.dataroot, mode='training', \
                                         batch_size=FLAGS.deep_all_batch_size, num_classes=FLAGS.num_classes,
                                         shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(
                tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

        test_data = ImageDataGenerator(test_file, dataroot=FLAGS.dataroot, mode='inference', \
                                       batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)
        test_next_batch = test_iterator.get_next()

        val_data = ImageDataGenerator(val_file, dataroot=FLAGS.dataroot, mode='inference', \
                                      batch_size=1, num_classes=FLAGS.num_classes, shuffle=False)
        val_iterator = tf.data.Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)
        val_next_batch = val_iterator.get_next()

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        train_batches_per_epoch.append(int(np.floor(tr_data_list[i].data_size / FLAGS.deep_all_batch_size)))

    test_init_op = test_iterator.make_initializer(test_data.data)
    test_batches_per_epoch = int(np.floor(test_data.data_size / 1))

    val_init_op = val_iterator.make_initializer(val_data.data)
    val_batches_per_epoch = int(np.floor(val_data.data_size / 1))

    # Training begins
    best_test_acc, best_val_acc = 0, 0
    for itr in range(resume_itr, FLAGS.train_iterations):

        # Sampling training and test tasks
        num_training_tasks = len(train_file_list)

        for i in range(len(train_file_list)):
            if itr % train_batches_per_epoch[i] == 0 or resume_itr != 0:
                sess.run(training_init_op[i])  # initialize training sample generator at itr=0

        # Sampling
        for i in range(num_training_tasks):
            task_ind = i
            if i == 0:
                inputa, labela = sess.run(train_next_list[task_ind])
            elif i == 1:
                inputa1, labela1 = sess.run(train_next_list[task_ind])
            else:
                inputb, labelb = sess.run(train_next_list[task_ind])
        feed_dict = {model.inputa: inputa, model.labela: labela, \
                     model.inputa1: inputa1, model.labela1: labela1, \
                     model.inputb: inputb, model.labelb: labelb, \
                     model.KEEP_PROB: 0.5}

        output_tensors = [model.task_train_op]
        output_tensors.extend([model.summ_op, model.source_loss, model.source_accuracy])
        _, summ_writer, source_loss, source_accuracy = sess.run(output_tensors, feed_dict)

        source_losses.append(source_loss)
        source_accuracies.append(source_accuracy)

        if itr % FLAGS.print_interval == 0:
            print('---' * 10 + '\n%s' % exp_string)
            print('Iteration %d' % itr + ': Loss ' + 'training domains ' + str(np.mean(source_losses)))
            print('Iteration %d' % itr + ': Accuracy ' + 'training domains ' + str(np.mean(source_accuracies)))
            # log loss and accuracy:
            path_save_train_acc = os.path.join(FLAGS.logdir, FLAGS.dataset, exp_string)
            if not os.path.exists(path_save_train_acc):
                os.makedirs(path_save_train_acc)
            with open(path_save_train_acc + '/eva_' + 'train' + '.txt', 'a') as fle:
                fle.write('Train results: Iteration %d, Loss: %f, Accuracy: %f \n' % (
                itr, np.mean(source_losses), np.mean(source_accuracies)))
            source_losses, target_losses = [], []

        if itr % FLAGS.summary_interval == 0 and FLAGS.log:
            train_writer.add_summary(summ_writer, itr)

        if itr % FLAGS.save_interval == 0:
            # saver.save(sess, FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + '/model' + str(itr))
            checkpoint_dir = FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + "/" + str(itr) + "/"
            save_network_model(saver_=saver, session_=sess, checkpoint_dir=checkpoint_dir,
                               model_name="model_itr" + str(itr))

        # Testing periodically
        if itr % FLAGS.val_print_interval == 0:
            val_acc, best_val_acc = evaluation_during_training(sess, model, exp_string, val_batches_per_epoch,
                                                               val_init_op, val_next_batch, itr, best_val_acc, saver,
                                                               is_val=True)
        if itr % FLAGS.test_print_interval == 0:
            test_acc, best_test_acc = evaluation_during_training(sess, model, exp_string, test_batches_per_epoch,
                                                                 test_init_op, test_next_batch, itr, best_test_acc,
                                                                 saver, is_val=False)


def evaluation_during_training(sess, model, exp_string, test_batches_per_epoch,
                               test_init_op, test_next_batch, itr, best_test_acc, saver, is_val=True):
    if FLAGS.masf_mode:
        method_ = "MASF"
    else:
        method_ = "DeepAll"
    if is_val:
        case_ = "val"
    else:
        case_ = "test"
    class_accs = [0.0] * FLAGS.num_classes
    class_samples = [0.0] * FLAGS.num_classes
    test_embeddings = np.zeros((test_batches_per_epoch, FLAGS.feature_space_dimension))
    test_labels = np.zeros((test_batches_per_epoch,))
    test_acc, test_loss, test_count = 0.0, 0.0, 0.0
    sess.run(test_init_op)  # initialize testing data generator
    for it in range(test_batches_per_epoch):
        test_input, test_label = sess.run(test_next_batch)
        feed_dict = {model.test_input: test_input, model.test_label: test_label, model.KEEP_PROB: 1.}
        if FLAGS.masf_mode:
            output_tensors = [model.test_loss, model.test_acc, model.semantic_feature, model.outputs,
                              model.metric_embedding]
        else:
            output_tensors = [model.test_loss, model.test_acc, model.embedding_feature, model.outputs]
        # --> model.semantic_feature: 4096-dimensional embedding with psi weights (for semantic features)
        # --> model.outputs: n_classes-dimensional embedding with theta weights (after softmax for cross-entropy)
        # --> model.metric_embedding: 256-dimensional embedding with phi weights (for metric features used in triplet loss)
        result = sess.run(output_tensors, feed_dict)
        test_loss += result[0]
        test_acc += result[1]
        test_count += 1
        this_class = np.argmax(test_label, axis=1)[0]
        class_accs[this_class] += result[1]  # added for debug
        class_samples[this_class] += 1
        test_embeddings[it, :] = result[2]
        test_labels[it] = np.argmax(test_label, axis=1)[0]
    test_acc = test_acc / test_count
    test_acc *= 100
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        # saver.save(sess, FLAGS.logdir + '/' + FLAGS.dataset + '/' + exp_string + '/itr' + str(itr) + '_model_acc' + str(best_test_acc))
    print('Unseen Target %s results: Iteration %d, Loss: %f, Accuracy: %f' % (
    case_, itr, test_loss / test_count, test_acc))
    if case_ == "test":
        print('Current best test accuracy {}'.format(best_test_acc))
    else:
        print('Current best val accuracy {}'.format(best_test_acc))
    # plot the test embedding:
    path_save = FLAGS.logdir + FLAGS.dataset + '/' + method_ + "/target_domain/plots/" + case_ + "/"
    _, _ = plot_embedding_of_points(embedding=test_embeddings, labels=test_labels, n_samples_plot=None)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    plt.savefig(path_save + 'epoch' + str(itr) + '.png')
    plt.clf()
    plt.close()
    # log loss and accuracy:
    with open((os.path.join(FLAGS.logdir, FLAGS.dataset, exp_string, 'eva_' + case_ + '.txt')), 'a') as fle:
        fle.write('Unseen Target %s results: Iteration %d, Loss: %f, Accuracy: %f \n' % (
        case_, itr, test_loss / test_count, test_acc))
    return test_acc, best_test_acc


def evaluation_after_training(model, sess, evaluation_file_list, evaluation_batch_size=1):
    if FLAGS.masf_mode:
        method_ = "MASF"
    else:
        method_ = "DeepAll"

    # Data loaders:
    with tf.device('/cpu:0'):
        evaluation_data_list, evaluation_iterator_list, evaluation_next_list = [], [], []
        for i in range(len(evaluation_file_list)):
            evaluation_data = ImageDataGenerator(evaluation_file_list[i], dataroot=FLAGS.dataroot, mode='inference', \
                                                 batch_size=evaluation_batch_size, num_classes=FLAGS.num_classes,
                                                 shuffle=False)
            evaluation_data_list.append(evaluation_data)
            evaluation_iterator_list.append(
                tf.data.Iterator.from_structure(evaluation_data.data.output_types, evaluation_data.data.output_shapes))
            evaluation_next_list.append(evaluation_iterator_list[i].get_next())

    # Ops for initializing different iterators:
    evaluation_init_op = []
    evaluation_batches_per_epoch = []
    for i in range(len(evaluation_file_list)):
        evaluation_init_op.append(evaluation_iterator_list[i].make_initializer(evaluation_data_list[i].data))
        evaluation_batches_per_epoch.append(int(np.floor(evaluation_data_list[i].data_size / evaluation_batch_size)))

    # Initialize iterator:
    num_total_domains = len(evaluation_file_list)
    for i in range(len(evaluation_file_list)):
        sess.run(evaluation_init_op[i])

        # Embedding:
    total_evaluation_embeddings = np.empty((0, FLAGS.feature_space_dimension))
    total_evaluation_labels_classwise = []
    total_evaluation_labels_domainwise = []
    total_evaluation_acc, total_evaluation_loss, total_evaluation_count = 0.0, 0.0, 0.0
    for domain_index in range(num_total_domains):
        class_accs = [0.0] * FLAGS.num_classes
        class_samples = [0.0] * FLAGS.num_classes
        evaluation_embeddings = np.zeros((evaluation_batches_per_epoch[domain_index], FLAGS.feature_space_dimension))
        evaluation_labels_classwise = np.zeros((evaluation_batches_per_epoch[domain_index],))
        evaluation_labels_domainwise = np.zeros((evaluation_batches_per_epoch[domain_index],))
        evaluation_acc, evaluation_loss, evaluation_count = 0.0, 0.0, 0.0
        for it in range(evaluation_batches_per_epoch[domain_index]):
            evaluation_input, evaluation_label_classwise = sess.run(evaluation_next_list[domain_index])
            feed_dict = {model.test_input: evaluation_input, model.test_label: evaluation_label_classwise,
                         model.KEEP_PROB: 1.}
            if FLAGS.masf_mode:
                output_tensors = [model.test_loss, model.test_acc, model.semantic_feature, model.outputs,
                                  model.metric_embedding]
            else:
                output_tensors = [model.test_loss, model.test_acc, model.embedding_feature, model.outputs]
            # --> model.semantic_feature: 4096-dimensional embedding with psi weights (for semantic features)
            # --> model.outputs: n_classes-dimensional embedding with theta weights (after softmax for cross-entropy)
            # --> model.metric_embedding: 256-dimensional embedding with phi weights (for metric features used in triplet loss)
            result = sess.run(output_tensors, feed_dict)
            evaluation_loss += result[0]
            total_evaluation_loss += result[0]
            evaluation_acc += result[1]
            total_evaluation_acc += result[1]
            evaluation_count += 1
            total_evaluation_count += 1
            this_class = np.argmax(evaluation_label_classwise, axis=1)[0]
            class_accs[this_class] += result[1]  # added for debug
            class_samples[this_class] += 1
            evaluation_embeddings[it, :] = result[2]
            evaluation_labels_classwise[it] = np.argmax(evaluation_label_classwise, axis=1)[0]
            evaluation_labels_domainwise[it] = domain_index
        evaluation_acc = evaluation_acc / evaluation_count
        evaluation_acc *= 100
        evaluation_loss = evaluation_loss / evaluation_count
        print('Evaluation results: Domain %d, Loss: %f, Accuracy: %f' % (domain_index, evaluation_loss, evaluation_acc))
        total_evaluation_embeddings = np.vstack((total_evaluation_embeddings, evaluation_embeddings))
        total_evaluation_labels_classwise.extend(evaluation_labels_classwise)
        total_evaluation_labels_domainwise.extend(evaluation_labels_domainwise)
        # save accuracy results:
        path_ = FLAGS.logdir + FLAGS.dataset + '/' + method_ + '/total_evaluation/accuracy/'
        if not os.path.exists(path_):
            os.makedirs(path_)
        with open((path_ + 'domain' + str(domain_index) + '_accuracy.txt'), 'a') as fle:
            fle.write('Evaluation results: Domain %d, Loss: %f, Accuracy: %f' % (
            domain_index, evaluation_loss, evaluation_acc))
    # save total average accuracy results (average over the whole evaulation data):
    total_evaluation_acc = total_evaluation_acc / total_evaluation_count
    total_evaluation_acc *= 100
    total_evaluation_loss_avg = total_evaluation_loss / total_evaluation_count
    path_ = FLAGS.logdir + FLAGS.dataset + '/' + method_ + '/total_evaluation/accuracy/'
    if not os.path.exists(path_):
        os.makedirs(path_)
    with open((path_ + "total_accuracy.txt"), 'a') as fle:
        fle.write('Evaluation results: Loss: %f, Accuracy: %f' % (total_evaluation_loss, total_evaluation_acc))

    # plot the classwise evaluation embedding:
    path_save1 = FLAGS.logdir + FLAGS.dataset + '/' + method_ + "/total_evaluation/plots/classwise/"
    path_save2 = FLAGS.logdir + FLAGS.dataset + '/' + method_ + "/total_evaluation/plots/domainwise/"
    name_save1, name_save2 = "embedding_classwise", "embedding_domainwise"
    plot_embedding_of_points_2PLOTS(embedding=total_evaluation_embeddings, labels1=total_evaluation_labels_classwise,
                                    labels2=total_evaluation_labels_domainwise,
                                    path_save1=path_save1, path_save2=path_save2,
                                    name_save1=name_save1, name_save2=name_save2, n_samples_plot=2000, method='TSNE')


def plot_embedding_of_points(embedding, labels, n_samples_plot=None, method='TSNE'):
    n_samples = embedding.shape[0]
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
    else:
        indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
    embedding_sampled = embedding[indices_to_plot, :]
    if embedding.shape[1] == 2:

        embedding_sampled = embedding
    else:
        if method == 'TSNE':
            embedding_sampled = TSNE(n_components=2).fit_transform(embedding_sampled)
        elif method == 'UMAP':
            embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
    labels_sampled = labels[indices_to_plot]
    _, ax = plt.subplots(1, figsize=(14, 10))
    n_classes = FLAGS.num_classes
    class_names = [class_list[str(i)] for i in range(len(class_list))]
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels_sampled, cmap='Spectral', alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(FLAGS.num_classes + 1) - 0.5)
    cbar.set_ticks(np.arange(FLAGS.num_classes))
    cbar.set_ticklabels(class_names)
    return plt, indices_to_plot


def plot_embedding_of_points_2PLOTS(embedding, labels1, labels2, path_save1, path_save2,
                                    name_save1, name_save2, n_samples_plot=None, method='TSNE'):
    n_samples = embedding.shape[0]
    if n_samples_plot != None:
        indices_to_plot = np.random.choice(range(n_samples), min(n_samples_plot, n_samples), replace=False)
    else:
        indices_to_plot = np.random.choice(range(n_samples), n_samples, replace=False)
    embedding_sampled = embedding[indices_to_plot, :]
    if embedding.shape[1] == 2:

        embedding_sampled = embedding
    else:
        if method == 'TSNE':
            embedding_sampled = TSNE(n_components=2).fit_transform(embedding_sampled)
        elif method == 'UMAP':
            embedding_sampled = umap.UMAP(n_neighbors=500).fit_transform(embedding_sampled)
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    labels1 = labels1.astype(int)
    labels2 = labels2.astype(int)
    labels1_sampled = labels1[indices_to_plot]
    labels2_sampled = labels2[indices_to_plot]
    #### plot1 (for classwise):
    _, ax = plt.subplots(1, figsize=(14, 10))
    n_classes = FLAGS.num_classes
    class_names = [class_list[str(i)] for i in range(len(class_list))]
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels1_sampled, cmap='Spectral', alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(FLAGS.num_classes + 1) - 0.5)
    cbar.set_ticks(np.arange(FLAGS.num_classes))
    cbar.set_ticklabels(class_names)
    if not os.path.exists(path_save1):
        os.makedirs(path_save1)
    plt.savefig(path_save1 + name_save1 + '.png')
    plt.clf()
    plt.close()
    #### plot2 (for domainwise):
    _, ax = plt.subplots(1, figsize=(14, 10))
    n_classes = len(domain_list)
    class_names = domain_list
    plt.scatter(embedding_sampled[:, 0], embedding_sampled[:, 1], s=10, c=labels2_sampled, cmap='Spectral', alpha=1.0)
    cbar = plt.colorbar(boundaries=np.arange(n_classes + 1) - 0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(class_names)
    if not os.path.exists(path_save2):
        os.makedirs(path_save2)
    plt.savefig(path_save2 + name_save2 + '.png')
    plt.clf()
    plt.close()
    np.save('embedding_sampled.npy', embedding_sampled)
    np.save('labels1_sampled.npy', labels1_sampled)
    np.save('labels2_sampled.npy', labels2_sampled)
    np.save('embedding.npy', embedding)
    np.save('labels1.npy', labels1)
    np.save('labels2.npy', labels2)


def save_network_model(saver_, session_, checkpoint_dir, model_name="model_"):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    # https://github.com/taki0112/ResNet-Tensorflow/blob/master/ResNet.py
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver_.save(session_, os.path.join(checkpoint_dir, model_name + '.model'))


def load_network_model(saver_, session_, checkpoint_dir):
    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver_.restore(session_, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False


if __name__ == "__main__":
    main()
