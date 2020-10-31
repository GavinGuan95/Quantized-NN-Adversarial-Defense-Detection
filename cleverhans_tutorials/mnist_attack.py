from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
import pickle

import time
import argparse
import logging
import os
import sys
from collections import OrderedDict

from cleverhans.utils import parse_model_settings, build_model_save_path
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval, tf_model_load, model_eval_multi_run

# The code block for enabling tensorflow gpu to work
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

FLAGS = flags.FLAGS

# Scaling input to softmax
INIT_T = 1.0
#ATTACK_T = 1.0
ATTACK_T = 0.25

# enum attack types
ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4

# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 2

MAX_BATCH_SIZE = 100

pickle_folder = "./pickle_result"
plt_folder = "./plot_result"

def mnist_attack(train_start=0, train_end=60000, test_start=0,
                 test_end=500, viz_enabled=True, nb_epochs=6,
                 batch_size=64, nb_filters=64,
                 nb_samples=10, learning_rate=0.001,
                 eps=0.3, attacks=[0],
                 attack_iterations=100, model_path=None,
                 targeted=False, binary=False, scale=False, rand=False,
                 debug=None, test=False,
                 data_dir=None, delay=0, adv=0, nb_iter=40, measure_uncertainty=True,
                 dropout_rate=0.0, bit=0):
    """
    MNIST tutorial for generic attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param nb_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1
    nb_classes = 10

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1237)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    if debug:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.WARNING)  # for running on sharcnet

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(datadir=data_dir,
                                                  train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    phase = tf.placeholder(tf.bool, name='phase')

    # for attempting to break unscaled network.
    logits_scalar = tf.placeholder_with_default(INIT_T, shape=(), name="logits_temperature")

    save = False
    train_from_scratch = False
    if model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                binary, scale, nb_filters, batch_size, learning_rate, nb_epochs, adv = parse_model_settings(model_path)
                train_from_scratch = False
            else:
                model_path = build_model_save_path(model_path, binary, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay, scale)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        train_from_scratch = True  # train from scratch, but don't save since no path given

    # Define TF model graph
    if binary:
        print('binary=True')
        if scale:
            print('scale=True')
            if rand:
                print('rand=True')
                from cleverhans_tutorials.tutorial_models import make_scaled_binary_rand_cnn
                model = make_scaled_binary_rand_cnn(phase, logits_scalar, 'binsc_', input_shape=(
                    None, img_rows, img_cols, channels), nb_filters=nb_filters, dropout=dropout_rate)
            else:
                from cleverhans_tutorials.tutorial_models import make_scaled_binary_cnn
                model = make_scaled_binary_cnn(phase, logits_scalar, 'binsc_', input_shape=(
                    None, img_rows, img_cols, channels), nb_filters=nb_filters, dropout=dropout_rate)
        else:
            if bit:
                from cleverhans_tutorials.tutorial_models import make_quantized_cnn
                model = make_quantized_cnn(phase, logits_scalar, 'bin_', nb_filters=nb_filters, dropout=dropout_rate, bit=bit)
            else:
                from cleverhans_tutorials.tutorial_models import make_basic_binary_cnn
                model = make_basic_binary_cnn(phase, logits_scalar, 'bin_', nb_filters=nb_filters, dropout=dropout_rate)

    else: # Non-binary models
        if rand:
            print('rand=True')
            from cleverhans_tutorials.tutorial_models import make_scaled_rand_cnn
            model = make_scaled_rand_cnn(phase, logits_scalar, 'fp_rand', nb_filters=nb_filters, dropout=dropout_rate)  # scaled uses SReLU
        else:
            from cleverhans_tutorials.tutorial_models import make_basic_cnn
            model = make_basic_cnn(phase, logits_scalar, 'fp_', nb_filters=nb_filters, dropout=dropout_rate)

    preds = model(x, reuse=False)  # * logits_scalar
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    rng = np.random.RandomState([2017, 8, 30])

    # Train an MNIST model
    train_params = {
        'binary': binary,
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_name': 'train loss',
        'filename': 'model',
        'reuse_global_step': False,
        'train_scope': 'train',
        'is_training': True
    }

    clean_tag = 'clean_'
    fgsm_tag = 'fgsm_'
    jsma_tag = 'jsma_'
    pgd_tag = 'pgd_'
    accuracy_tag = 'accuracy'
    group_accuracy_tag = 'group_accuracy'
    uncertainty_list_tag = 'uncertainty_list'
    uncertainty_mean_tag = 'uncertainty_mean'
    uncertainty_var_tag = 'uncertainty_var'
    lr_tag = 'lr'

    # collect results of the attacks/defenses, write the result to a pickle file
    result_collection = OrderedDict([
        ('binary', binary),
        ('rand', rand),
        ('dropout', dropout_rate),
        ('attack', attacks),
        ('adv_tr', adv),
        ('bit', bit),
        ('eps', eps),
        ('clean_accuracy', None),
        ('clean_group_accuracy', None),
        ('clean_uncertainty_list', None),
        ('clean_uncertainty_mean', None),
        ('clean_uncertainty_var', None),
        ('fgsm_accuracy', None),
        ('fgsm_group_accuracy', None),
        ('fgsm_uncertainty_list', None),
        ('fgsm_uncertainty_mean', None),
        ('fgsm_uncertainty_var', None),
        ('fgsm_lr', None),
        ('jsma_accuracy', None),
        ('jsma_group_accuracy', None),
        ('jsma_uncertainty_list', None),
        ('jsma_uncertainty_mean', None),
        ('jsma_uncertainty_var', None),
        ('jsma_lr', None),
        ('pgd_accuracy', None),
        ('pgd_group_accuracy', None),
        ('pgd_uncertainty_list', None),
        ('pgd_uncertainty_mean', None),
        ('pgd_uncertainty_var', None),
        ('pgd_lr', None),
    ])

    if adv != 0:
        if adv == ADVERSARIAL_TRAINING_MADRYETAL:
            from cleverhans.attacks import MadryEtAl
            train_attack_params = {'eps': MAX_EPS, 'eps_iter': 0.01, 'nb_iter': nb_iter}
            train_attacker = MadryEtAl(model, sess=sess)
        elif adv == ADVERSARIAL_TRAINING_FGSM:
            from cleverhans.attacks import FastGradientMethod
            stddev = int(np.ceil((MAX_EPS * 255) // 2))
            train_attack_params = {'eps': tf.abs(tf.truncated_normal(shape=(batch_size, 1, 1, 1), mean=0, stddev=stddev))}
            train_attacker = FastGradientMethod(model, back='tf', sess=sess)

        # create the adversarial trainer
        train_attack_params.update({'clip_min': 0., 'clip_max': 1.})
        adv_x_train = train_attacker.generate(x, phase, **train_attack_params)
        preds_adv_train = model.get_probs(adv_x_train)

        eval_attack_params = {'eps': MAX_EPS, 'clip_min': 0., 'clip_max': 1.}
        adv_x_eval = train_attacker.generate(x, phase, **eval_attack_params)
        preds_adv_eval = model.get_probs(adv_x_eval)  # * logits_scalar

    def evaluate():
        # Evaluate the accuracy of the MNIST model on clean test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

        if adv != 0:
            # Accuracy of the adversarially trained model on adversarial
            # examples
            acc = model_eval(sess, x, y, preds_adv_eval, X_test, Y_test, phase=phase, args=eval_params)
            print('Test accuracy on adversarial examples: %0.4f' % acc)

            acc = model_eval(sess, x, y, preds_adv_eval, X_test, Y_test, phase=phase, args=eval_params, feed={logits_scalar: ATTACK_T})
            print('Test accuracy on adversarial examples (scaled): %0.4f' % acc)

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

        # do clean training for 'nb_epochs' or 'delay' epochs
        if test:
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase, evaluate=evaluate, args=train_params, save=save, rng=rng)
        else:
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            if test:
                model_train(sess, x, y, preds, X_train, Y_train, phase=phase, predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params, save=save, rng=rng)
            else:
                model_train(sess, x, y, preds, X_train, Y_train, phase=phase, predictions_adv=preds_adv_train, args=train_params, save=save, rng=rng)
    else:
        tf_model_load(sess, model_path)
        print('Restored model from %s' % model_path)
        evaluate()

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    # Clean uncertainty measured with test examples
    if measure_uncertainty:
        clean_accuracy, mean_var_clean, square_mean_var_clean, group_accuracy_clean = model_eval_multi_run(sess, x, y, preds, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
    else:
        clean_accuracy = model_eval(sess, x, y, preds, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
    if measure_uncertainty:
        print("Clean test, acc: {}, group acc: {}".format(clean_accuracy, group_accuracy_clean))
    else:
        print("Clean test, acc: {}".format(clean_accuracy))

    result_collection[clean_tag+accuracy_tag] = clean_accuracy
    if measure_uncertainty:
        result_collection[clean_tag+group_accuracy_tag] = group_accuracy_clean
        result_collection[clean_tag+uncertainty_list_tag] = mean_var_clean
        result_collection[clean_tag+uncertainty_mean_tag] = np.mean(mean_var_clean)
        result_collection[clean_tag+uncertainty_var_tag] = np.var(mean_var_clean)

    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(clean_accuracy))
    report.clean_train_clean_eval = clean_accuracy

    ###########################################################################
    # Build dataset
    ###########################################################################
    if viz_enabled:
        assert nb_samples == nb_classes
        idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
                for i in range(nb_classes)]
        viz_rows = nb_classes if targeted else 2
        # Initialize our array for grid visualization
        grid_shape = (nb_classes, viz_rows, img_rows, img_cols, channels)
        grid_viz_data = np.zeros(grid_shape, dtype='f')

    if targeted:
        from cleverhans.utils import build_targeted_dataset
        if viz_enabled:
            from cleverhans.utils import grid_visual
            adv_inputs, true_labels, adv_ys = build_targeted_dataset(X_test, Y_test, idxs, nb_classes, img_rows, img_cols, channels)
        else:
            adv_inputs, true_labels, adv_ys = build_targeted_dataset(X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
    else:
        if viz_enabled:
            from cleverhans.utils import pair_visual
            adv_inputs = X_test[idxs]
        else:
            adv_inputs = X_test[:nb_samples]

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    if targeted:
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"

    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) + ' adversarial examples')
    print("This could take some time ...")

    attacker_dict = {}

    for attack in attacks:
        if attack == ATTACK_CARLINI_WAGNER_L2:
            print('Attack: CarliniWagnerL2')
            from cleverhans.attacks import CarliniWagnerL2
            attacker = CarliniWagnerL2(model, back='tf', sess=sess)
            attack_params = {'binary_search_steps': 1,
                             'max_iterations': attack_iterations,
                             'learning_rate': 0.1,
                             'batch_size': att_batch_size,
                             'initial_const': 10,
                             }
            attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
        elif attack == ATTACK_JSMA:
            print('Attack: SaliencyMapMethod')
            from cleverhans.attacks import SaliencyMapMethod
            attacker = SaliencyMapMethod(model, back='tf', sess=sess)
            attack_params = {'theta': 1., 'gamma': 0.1}
            attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
        elif attack == ATTACK_FGSM:
            print('Attack: FastGradientMethod')
            from cleverhans.attacks import FastGradientMethod
            attacker = FastGradientMethod(model, back='tf', sess=sess)
            attack_params = {'eps': eps}
            attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
        elif attack == ATTACK_MADRYETAL:
            print('Attack: MadryEtAl')
            from cleverhans.attacks import MadryEtAl
            attacker = MadryEtAl(model, back='tf', sess=sess)
            attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
            attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
        elif attack == ATTACK_BASICITER:
            print('Attack: BasicIterativeMethod')
            from cleverhans.attacks import BasicIterativeMethod
            attacker = BasicIterativeMethod(model, back='tf', sess=sess)
            attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
            attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
        else:
            print("Attack undefined")
            sys.exit(1)
        attacker_dict[attack] = (attacker, attack_params)

    adv_np_dict = {}
    for att_key, att_val in attacker_dict.items():
        attacker = att_val[0]
        attack_params = att_val[1]
        adv_np = attacker.generate_np(adv_inputs, phase, **attack_params)
        adv_np_dict[att_key] = adv_np

    for att_key, adv_np in adv_np_dict.items():
        eval_params = {'batch_size': att_batch_size}
        print("Evaluating untargeted results (Adversarial input accuracy evaluation)")
        if measure_uncertainty:
            adv_accuracy, mean_var, square_mean_var, group_accuracy_adv = model_eval_multi_run(sess, x, y, preds, adv_np, Y_test[:nb_samples], phase=phase, args=eval_params)
        else:
            adv_accuracy = model_eval(sess, x, y, preds, adv_np, Y_test[:nb_samples], phase=phase, args=eval_params)
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression

        # perform a logistic regression
        if measure_uncertainty:
            combined_var = np.concatenate((mean_var, mean_var_clean), axis=0)
            combined_var = np.reshape(combined_var, (-1, 1))
            combined_clean_or_adv = np.concatenate((np.zeros(mean_var_clean.shape[0]), np.ones(mean_var.shape[0])), axis=0)
            LR = LogisticRegression(random_state=0, solver='lbfgs').fit(combined_var, combined_clean_or_adv)
            lr_score = LR.score(combined_var, combined_clean_or_adv)

        if att_key == ATTACK_FGSM:
            tag = fgsm_tag
        elif att_key == ATTACK_JSMA:
            tag = jsma_tag
        elif att_key == ATTACK_MADRYETAL:
            tag = pgd_tag

        result_collection[tag+accuracy_tag] = adv_accuracy
        if measure_uncertainty:
            result_collection[tag+group_accuracy_tag] = group_accuracy_adv
            result_collection[tag+uncertainty_list_tag] = mean_var
            result_collection[tag+uncertainty_mean_tag] = np.mean(mean_var)
            result_collection[tag+uncertainty_var_tag] = np.var(mean_var)
            result_collection[tag+lr_tag] = lr_score


    # Close TF session
    sess.close()

    # save result collection to pickle file
    info_list = ['binary', 'rand', 'dropout', 'attack', 'adv_tr', 'bit', 'eps']
    pickle_file_name = ""
    for info in info_list:
        pickle_file_name += info + "_" + str(result_collection[info]) + "_"
        valid_char = '-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        pickle_file_name = "".join([x if x in valid_char else "_" for x in pickle_file_name])
    pickle_file_name = pickle_file_name.replace("__", "_")
    pickle_file_name = pickle_file_name.rstrip("_")
    pickle_file_name += ".pickle"
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    pickle_filepath = os.path.join(pickle_folder, pickle_file_name)
    with open(pickle_filepath, "wb") as f:
        pickle.dump(result_collection, f)
        print("Result of this run has been written to: {}".format(pickle_filepath))

    # plot histogram for variance
    if measure_uncertainty:
        for att_key in adv_np_dict.keys():
            info_list = ['binary', 'rand', 'dropout', 'adv_tr', 'bit', 'eps']
            plt_file_name = ""
            for info in info_list:
                plt_file_name += info + "_" + str(result_collection[info]) + "_"
                valid_char = '-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                plt_file_name = "".join([x if x in valid_char else "_" for x in plt_file_name])
            histogram_bins = np.linspace(0, 0.035, 100)
            if att_key == ATTACK_FGSM:
                tag = fgsm_tag
            elif att_key == ATTACK_JSMA:
                tag = jsma_tag
            elif att_key == ATTACK_MADRYETAL:
                tag = pgd_tag
            plt_file_name += "_" + tag
            mean_var = result_collection[tag+uncertainty_list_tag]
            plt.figure(figsize=(16.0, 10.0))
            plt.hist(mean_var, histogram_bins, alpha=0.5, label='Adversarial Input Variance', color='blue')
            plt.hist(mean_var_clean, histogram_bins, alpha=0.5, label='Clean Input Variance', color='green')
            plt.legend(loc='upper right')
            if not os.path.exists(plt_folder):
                os.makedirs(plt_folder)
            plt.savefig(os.path.join(plt_folder, plt_file_name), bbox_inches='tight')

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_attack(viz_enabled=FLAGS.viz_enabled,
                 nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 nb_samples=FLAGS.nb_samples,
                 nb_filters=FLAGS.nb_filters,
                 learning_rate=FLAGS.lr,
                 eps=FLAGS.eps,
                 attacks=FLAGS.attack,
                 attack_iterations=FLAGS.attack_iterations,
                 model_path=FLAGS.model_path,
                 targeted=FLAGS.targeted,
                 binary=FLAGS.binary,
                 scale=FLAGS.scale,
                 rand=FLAGS.rand,
                 debug=FLAGS.debug,
                 test=FLAGS.test,
                 data_dir=FLAGS.data_dir,
                 delay=FLAGS.delay,
                 adv=FLAGS.adv,
                 nb_iter=FLAGS.nb_iter,
                 measure_uncertainty=FLAGS.measure_uncertainty,
                 dropout_rate=FLAGS.dropout,
                 bit=FLAGS.bit)


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
    par.add_argument('--data_dir', help='Path to training data',
                     default='/tmp/mnist')
    par.add_argument(
        '--viz_enabled', help='Visualize adversarial ex.', action="store_true")
    par.add_argument(
        '--debug', help='Sets log level to DEBUG, otherwise INFO', action="store_true")
    par.add_argument(
        '--test', help='Test while training, takes longer', action="store_true")

    # Architecture and training specific flags
    par.add_argument('--nb_epochs', type=int, default=15,
                     help='Number of epochs to train model')
    par.add_argument('--nb_filters', type=int, default=64,
                     help='Number of filters in first layer')
    par.add_argument('--batch_size', type=int, default=64,
                     help='Size of training batches')
    par.add_argument('--lr', type=float, default=0.001,
                     help='Learning rate')
    par.add_argument('--binary', help='Use a binary model?',
                     action="store_true")
    par.add_argument('--scale', help='Scale activations after binarization or sampling weights?',
                     action="store_true")
    par.add_argument('--rand', help='Stochastic weight layer if set',
                     action="store_true")

    # Attack specific flags
    par.add_argument('--attack', type=int, nargs='+', default=0,
                     help='Attack type, 0=CW, 1=JSMA, 2=FGSM, 3=MADRYETAL')
    par.add_argument("--eps", type=float, default=0.3)
    par.add_argument('--attack_iterations', type=int, default=100,
                     help='Number of iterations to run CW attack; 100 is good')  # The attack iterations for CW, just keep it as 100
    par.add_argument('--nb_samples', type=int,
                     default=10, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")

    # Adversarial training flags
    par.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int, default=40, help='Nb of iterations of PGD')
    par.add_argument('--measure_uncertainty', help='Measure Uncertainty of Model through multiple runs', action="store_true")
    par.add_argument('--dropout', type=float, default=0.0)
    par.add_argument('--bit', type=int, default=0)
    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
