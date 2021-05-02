# this code compute memory needed for mobilenet v2 theoretically

import numpy as np

# memory for weight parameters
param_mem_layers = []
# memory for activation stored for learning
act_mem_layers = []

im2col = True

##########
# function for compute memory for one batch
#   return param_memory, act_memory
##########
def mem_for_linear(input_size, output_size):
  return input_size * output_size, input_size

def mem_for_conv(input_size, output_size, ipc, opc, kernel_size):
  # im2col conv
  if (im2col):
    return ipc * opc * kernel_size * kernel_size, output_size * output_size * ipc * kernel_size * kernel_size
  else:
    return ipc * opc * kernel_size * kernel_size, output_size * output_size * ipc

def mem_for_batchnorm():

def mem_for_activation(input_size, ipc):
  return 0, input_size * input_size * ipc

def mem_for_inverted_residual(ipc, opc, stride, expand_ratio):
  hidden_dim = int(ipc * expand_ratio)

def mem_for_mobilenetv2():

##########
# main function
##########
def main():




if __name__ == "__main__":
  main()