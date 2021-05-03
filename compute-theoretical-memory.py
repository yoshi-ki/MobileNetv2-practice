# this code compute memory needed for mobilenet v2 theoretically
import numpy as np
import math

# configuration for this code
im2col = True
batch_size = 1


# memory for weight parameters
param_mem_layers = []
# memory for activation stored for learning
act_mem_layers = []


def _make_divisible(v, divisor, min_value=None):
  """
  This function is taken from the original tf repo.
  It ensures that all layers have a channel number that is divisible by 8
  It can be seen here:
  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
  :param v:
  :param divisor:
  :param min_value:
  :return:
  """
  if min_value is None:
      min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
      new_v += divisor
  return new_v


##########
# basic block function to compute memory for mini-batch
#   return output_size, param_memory, act_memory
##########
def mem_for_linear(input_size, output_size):
  return output_size, input_size * output_size, input_size * batch_size

def mem_for_conv(input_size, ipc, opc, kernel_size, stride=1, padding=0):
  output_size = math.floor(((input_size + 2 * padding - (kernel_size - 1) - 1) / stride) + 1)
  # im2col conv or not
  if (im2col):
    return output_size, ipc * opc * kernel_size * kernel_size, output_size * output_size * ipc * kernel_size * kernel_size * batch_size
  else:
    return output_size, ipc * opc * kernel_size * kernel_size, output_size * output_size * ipc * batch_size

def mem_for_batchnorm(input_size, ipc):
  output_size = input_size
  if batch_size == 1:
    return output_size, 0, 0
  else:
    # store weight for gamma and beta
    # store activation for E[x] and V[x]
    return output_size, 2, input_size * input_size * ipc + 1

def mem_for_activation_function(input_size, ipc):
  output_size = input_size
  return output_size, 0, input_size * input_size * ipc * batch_size




##########
# function to compute memory for mini-batch
#   return output_size, param_memory, act_memory
##########
def mem_for_conv_1x1_bn(input_size, ipc, opc):
  weight_mem_all, act_mem_all = 0, 0
  output_size, weight_mem, act_mem = mem_for_conv(input_size, ipc, opc, 1)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  output_size, weight_mem, act_mem = mem_for_batchnorm(output_size, opc)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  output_size, weight_mem, act_mem = mem_for_activation_function(output_size, opc)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  return output_size, weight_mem_all, act_mem_all

def mem_for_conv_3x3_bn(input_size, ipc, opc, stride):
  weight_mem_all, act_mem_all = 0, 0
  output_size, weight_mem, act_mem = mem_for_conv(input_size, ipc, opc, 3, stride, 1)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  output_size, weight_mem, act_mem = mem_for_batchnorm(output_size, opc)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  output_size, weight_mem, act_mem = mem_for_activation_function(output_size, opc)
  weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
  return output_size, weight_mem_all, act_mem_all


def mem_for_inverted_residual(input_size, ipc, opc, stride, expand_ratio):
  weight_mem_all = 0
  act_mem_all = 0
  hidden_dim = int(ipc * expand_ratio)
  if (expand_ratio == 1):
    #1st
    output_size, weight_mem, act_mem = mem_for_conv(input_size, hidden_dim, hidden_dim, 3, stride, 1)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_batchnorm(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_activation_function(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    # 2nd
    output_size, weight_mem, act_mem = mem_for_conv(input_size, hidden_dim, opc, 1, 1, 0)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_batchnorm(input_size, opc)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    return output_size, weight_mem_all, act_mem_all
  else:
    # 1st
    output_size, weight_mem, act_mem = mem_for_conv(input_size, ipc, hidden_dim, 1)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_batchnorm(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_activation_function(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    # 2nd
    output_size, weight_mem, act_mem = mem_for_conv(input_size, hidden_dim, hidden_dim, 3, stride, 1)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_batchnorm(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_activation_function(input_size, hidden_dim)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    # 3rd
    output_size, weight_mem, act_mem = mem_for_conv(input_size, hidden_dim, opc, 1)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    output_size, weight_mem, act_mem = mem_for_batchnorm(input_size, opc)
    weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
    input_size = output_size
    return output_size, weight_mem_all, act_mem_all


def mem_for_mobilenetv2(n_class=10, mult_width_param=1):
  # variables and configurations
  input_size = 224
  weight_mem_all = 0
  act_mem_all = 0
  net_configs = [
    # t, c, n, s
    [1,  16, 1, 1],
    [6,  24, 2, 2],
    [6,  32, 3, 2],
    [6,  64, 4, 2],
    [6,  96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
  ]
  # first layer
  input_channel = 32
  output_size, weight_mem, act_mem = mem_for_conv_3x3_bn(input_size, 3, input_channel, 2)
  weight_mem_all += weight_mem
  act_mem_all    += act_mem
  input_size      = output_size
  # bottleneck blocks
  for t, c, n, s in net_configs:
    output_channel = _make_divisible(c * mult_width_param, 4 if mult_width_param == 0.1 else 8)
    for i in range(n):
      output_size, weight_mem, act_mem = mem_for_inverted_residual(input_size, input_channel, output_channel, s if i == 0 else 1, t)
      weight_mem_all, act_mem_all = weight_mem_all + weight_mem, act_mem_all + act_mem
      input_size = output_size
      input_channel = output_channel
  # average pooling is no memory
  # last classifier
  output_channel = _make_divisible(1280 * mult_width_param, 4 if mult_width_param == 0.1 else 8) if mult_width_param > 1.0 else 1280

  _, weight_mem, act_mem= mem_for_linear(output_channel, n_class)
  weight_mem_all += weight_mem
  act_mem_all    += act_mem
  input_size      = output_size

  print("overall memory for mobilenet-v2")
  print("weight: %d, activation: %d, ratio (weight/activation): %3f" % (weight_mem_all, act_mem_all, weight_mem_all/act_mem_all))



if __name__ == "__main__":
  mem_for_mobilenetv2()