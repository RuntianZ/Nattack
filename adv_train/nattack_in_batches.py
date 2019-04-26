import keras
import keras.backend as K
import os
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import argparse
import tensorflow as tf
import numpy as np


def torch_arctanh(x, eps=1e-6):
  x *= (1. - eps)
  return (np.log((1 + x) / (1 - x))) * 0.5


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PGD Attack')
  parser.add_argument('--ckpt', default='cifar10_normal.h5', type=str, help='Checkpoint path')
  parser.add_argument('--eps', default=8.0/255.0, type=float)
  parser.add_argument('--step_num', default=100, type=int, help='Number of attack trials')
  parser.add_argument('--npop', default=100, type=int)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--sigma', default=0.1, type=float)
  parser.add_argument('--alpha', default=0.008, type=float)
  args = parser.parse_args()

  boxmin = 0
  boxmax = 1
  boxplus = (boxmin + boxmax) / 2.
  boxmul = (boxmax - boxmin) / 2.

  # Prepare data
  num_classes = 10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.astype('float32') / 255.0
  x_test = x_test.astype('float32') / 255.0
  logits_train = keras.utils.to_categorical(y_train, num_classes)
  logits_test = keras.utils.to_categorical(y_test, num_classes)
  y_train = y_train.squeeze()
  y_test = y_test.squeeze()

  gen_train = ImageDataGenerator()
  gen_train.fit(x_train)
  gen_test = ImageDataGenerator()
  gen_test.fit(x_test)

  # Prepare model
  model = keras.models.load_model(args.ckpt)
  input_xs = model.input
  output_ys = model.output
  targets_ys = K.placeholder([None, 10], dtype=tf.float32)
  loss_t = keras.losses.categorical_crossentropy(output_ys, targets_ys)
  grad_t = K.gradients(loss_t, input_xs)[0]

  # Attack training set
  for (x_batch, y_batch) in gen_train.flow(x_train, y_train, batch_size=args.batch_size):

    mask = np.ones((len(x_batch),), dtype=int)
    prediction = np.argmax(model.predict(x_batch), axis=1)
    mask &= (prediction == y_batch)

    modify = np.random.randn(len(x_batch), 32, 32, 3) * 0.001
    x_old_batch = np.tile(x_batch, (args.npop, 1, 1, 1))
    y_batch_tile = np.tile(y_batch, (args.npop,))
    logits_batch_tile = keras.utils.to_categorical(y_batch_tile, num_classes)

    for runstep in range(args.step_num):
      Nsample = np.random.randn(args.npop, 32, 32, 3)
      Nsample_batch = np.tile(Nsample, (len(x_batch), 1, 1, 1))
      modify_try = modify.repeat(args.npop, 0) + args.sigma * Nsample_batch

      x_new_batch = np.tile(torch_arctanh((x_batch - boxplus) / boxmul), (args.npop, 1, 1, 1))
      x_new_batch = np.tanh(x_new_batch + modify_try) * boxmul + boxplus
      x_new_batch = np.clip(x_new_batch, x_old_batch - args.eps, x_old_batch + args.eps)
      x_new_batch = np.clip(x_new_batch, 0.0, 1.0)

      outputs = model.predict(x_new_batch)
      prediction = np.argmax(outputs, axis=1)

      real = np.log((logits_batch_tile * outputs).sum(1) + 1e-30)
      other = np.log(((1. - logits_batch_tile) * outputs - logits_batch_tile * 10000.).max(1)[0] + 1e-30)
      loss1 = np.clip(real - other, 0., 1000.)
      reward = -0.5 * loss1
      A = (reward - np.mean(reward)) / (np.std(reward) + 1e-7)
      modify = modify + (args.alpha / (args.npop * args.sigma)) * ((np.dot(Nsample_batch.reshape(len(x_batch) * args.npop, -1).T, A)).reshape(32, 32, 3))

      # Test
      if runstep % 10 == 9:
        x_test_batch = torch_arctanh((x_batch - boxplus) / boxmul)
        x_test_batch = np.tanh(x_test_batch + modify) * boxmul + boxplus
        x_test_batch = np.clip(x_test_batch, x_batch - args.eps, x_batch + args.eps)
        x_test_batch = np.clip(x_test_batch, 0.0, 1.0)
        l2real = np.sum((x_test_batch - x_batch) ** 2) ** 0.5
        print('l2real:', l2real)
        prediction = np.argmax(model.predict(x_test_batch), axis=1)
        mask &= (prediction == y_batch)
        print(np.sum(mask))
        if np.sum(mask) == 0:
          break


    print('Survival: {} of {}'.format(np.sum(mask), len(x_batch)))
