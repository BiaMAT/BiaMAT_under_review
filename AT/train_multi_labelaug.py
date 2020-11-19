"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
import sys
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
import math

from model import Model
import cifar10_input
import imagenet32_input
from pgd_attack import LinfPGDAttack
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', type=str, help='suffix')
parser.add_argument('--shuffle', help='shuffling test', action='store_true')
parser.add_argument('--n', type=int, default=4)
parser.add_argument('--alpha', type=float, help='coefficient for the auxiliary head')
args = parser.parse_args()

conf = 'config.json'
with open(conf) as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
data_path_aux = config['data_path_aux']
momentum = config['momentum']
batch_size = config['training_batch_size']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

args.alpha
num_classes = 10

raw_auxs = imagenet32_input.ImageNetData(data_path_aux, args.shuffle)
raw_cifar = cifar10_input.CIFAR10Data(data_path)

# Setting up the data and the model
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(num_classes, aux_num_classes=1000, n=args.n)

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent_merge + weight_decay * model.weight_decay_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
        total_loss,
        global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       'ssv_train')

attack_eval = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir'] + args.suffix
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=1)
tf.summary.scalar('accuracy adv train cifar10', model.accuracy)
#tf.summary.scalar('accuracy adv', model.accuracy)
#tf.summary.scalar('xent adv train', model.xent / batch_size)
#tf.summary.scalar('xent adv', model.xent / batch_size)
#tf.summary.image('images adv train', model.x_input, max_outputs=128)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy(conf, model_dir)

tops = [0]
max_acc_adv = 0
acc_nat_at_max_adv = 0
def comp_and_save_ckpt(tops, new_adv, new_nat, summary_writer, sess, global_step):
  m = min(tops)
  if m <= new_adv:
    tops.remove(m)
    tops.append(new_adv)
    saver.save(sess,
               os.path.join(model_dir, 'checkpoint'),
               global_step=global_step)
    summary_comp = tf.Summary(value=[
          tf.Summary.Value(tag='max acc adv', simple_value= max(tops)),
          tf.Summary.Value(tag='acc nat at max adv', simple_value= new_nat)])
    summary_writer.add_summary(summary_comp, global_step.eval(sess))
  return tops

def evaluate(sess):
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr_nat = 0
  total_corr_adv = 0
  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)

    x_batch = raw_cifar.eval_data.xs[bstart:bend, :]
    y_batch = raw_cifar.eval_data.ys[bstart:bend]

    x_batch_adv = attack_eval.perturb(x_batch, y_batch, sess, alpha=1.0, is_training=False)

    dict_nat = {model.x_input: x_batch,
                model.is_training: False,
                model.y_input: y_batch
                }

    dict_adv = {model.x_input: x_batch_adv,
                model.is_training: False,
                model.y_input: y_batch
                }

    cur_corr_nat = sess.run(model.num_correct,feed_dict = dict_nat)
    cur_corr_adv = sess.run(model.num_correct,feed_dict = dict_adv)

    total_corr_nat += cur_corr_nat
    total_corr_adv += cur_corr_adv

  acc_nat = total_corr_nat / num_eval_examples
  acc_adv = total_corr_adv / num_eval_examples

  summary_eval = tf.Summary(value=[
        tf.Summary.Value(tag='acc nat', simple_value= acc_nat),
        tf.Summary.Value(tag='acc adv', simple_value= acc_adv)])
  return summary_eval, acc_adv, acc_nat

with tf.Session() as sess:
  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
  aux = imagenet32_input.AugmentedImageNetData(raw_auxs, sess, model)
  sess.run(tf.global_variables_initializer())

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  training_time = 0.0
  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch10, y_batch10 = cifar.train_data.get_next_batch(batch_size // 2,
                                                       multiple_passes=True)
    x_batch_aux, y_batch_aux = aux.train_data.get_next_batch(batch_size // 2, multiple_passes=True)
    x_batch = np.concatenate([x_batch10, x_batch_aux], axis=0)
    y_batch = np.concatenate([y_batch10, y_batch_aux])

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess, alpha=args.alpha, is_training=True)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.is_training: True,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.is_training: True,
                model.alpha: args.alpha,
                model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start

    # Tensorboard summaries
    if (ii == 0) or ((ii+1) % num_summary_steps == 0):
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))
      summary_eval, acc_adv, acc_nat = evaluate(sess)
      summary_writer.add_summary(summary_eval, global_step.eval(sess))
      if ii > step_size_schedule[1][0]:
        tops = comp_and_save_ckpt(tops, acc_adv, acc_nat, summary_writer, sess, global_step)
