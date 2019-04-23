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
  parser.add_argument('--ckpt', type=str, help='Checkpoint path')
  parser.add_argument('--eps', default=8.0/255.0, type=float)
  parser.add_argument('--step_num', default=40, type=int, help='Number of attack trials')
  parser.add_argument('--npop', default=100, type=int)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--sigma', default=0.1, type=float)
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

    for runstep in range(args.step_num):
      Nsample = np.random.randn(args.npop, 32, 32, 3)
      Nsample_batch = np.tile(Nsample, (len(x_batch), 1, 1, 1))
      modify_try = modify.repeat(args.npop, 0) + args.sigma * Nsample_batch

      x_new_batch = np.tile(torch_arctanh((x_batch - boxplus) / boxmul), (args.npop, 1, 1, 1))
      x_new_batch = np.tanh(x_new_batch + modify_try) * boxmul + boxplus
      x_new_batch = np.clip(x_new_batch, x_old_batch - args.eps, x_old_batch + args.eps)
      x_new_batch = np.clip(x_new_batch, 0.0, 1.0)

      prediction = np.argmax(model.predict(x_new_batch), axis=1)

      for i in range(args.npop):
        mask &= (prediction[i * len(x_batch):(i + 1) * len(x_batch)] == y_batch)

    print('Survival: {} of {}'.format(np.sum(mask), len(x_batch)))
