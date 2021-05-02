# essential libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import os
# hydra
import hydra
from omegaconf import DictConfig, OmegaConf

from mobilenetv2 import MobileNetv2
from utils import *


def calculate_accuracy(output, target):
  correct = 0
  total = 0
  _, predicted = torch.max(output.data, 1)
  total += target.size(0)
  correct += (predicted==target).sum().item()
  accuracy = (100 * float(correct/total))
  return accuracy

@hydra.main(config_name="config")
def main(cfg: DictConfig) -> None:
  # configuration
  model_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.result_config.model_dir)
  graph_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.result_config.graph_dir)
  mkdir(model_dir)
  mkdir(graph_dir)

  # load datasets
  data_root = cfg.dataset_config.dir
  preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  dataset_train = datasets.CIFAR10(root=data_root,train=True,download=True,transform=preprocess)
  dataset_test = datasets.CIFAR10(root=data_root,train=False,download=True,transform=preprocess)
  dataloader_train = DataLoader(dataset_train, batch_size=cfg.train_config.batch_size, shuffle=True, num_workers=16)
  dataloader_test = DataLoader(dataset_test, batch_size=cfg.train_config.batch_size, shuffle=False, num_workers=16)

  # define network, loss function, and optimizer
  net = MobileNetv2()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.RMSprop(net.parameters())

  # device configuration
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print('using device:', device)
  net = net.to(device)

  # train loop
  epochs = cfg.train_config.epochs
  train_acc_list = np.zeros(epochs)
  test_acc_list = np.zeros(epochs)
  test_acc_best = 0

  for epoch in range(epochs):
    # train
    tic = time.time()
    train_acc = 0
    count = 0
    for i, (inputs, target) in enumerate(dataloader_train):
      inputs = inputs.to(device)
      optimizer.zero_grad()
      outputs = net(inputs)
      outputs = outputs.to("cpu")
      loss = criterion(outputs, target)
      loss.backward()
      optimizer.step()
      train_acc += calculate_accuracy(outputs, target)
      count += 1
    train_acc /= count
    train_acc_list[epoch] = train_acc

    # test
    test_acc = 0
    count = 0
    for i, (inputs, target) in enumerate(dataloader_test):
      with torch.no_grad():
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = outputs.to("cpu")
        test_acc += calculate_accuracy(outputs, target)
        count += 1
    test_acc /= count
    test_acc_list[epoch] = test_acc

    # if accuracy is the best save model.
    if (test_acc_best < test_acc):
      print("best accuracy recorded")
      test_acc_best = test_acc
      torch.save(net.state_dict(), model_dir + "cifar10_best.pt")

    # print results for each epoch
    toc = time.time()
    print("epoch %d/%d, train_accuracy = %f, test_accuracy = %f, " % (
    epoch, epochs, train_acc_list[epoch], test_acc_list[epoch]), end="")
    print("time: %f seconds\n" % (toc - tic))

    # sometimes draw graph
    freq = 2
    if(epoch % freq == 0):
      # draw graph
      textsize = 15
      marker = 5
      plt.figure(dpi=100)
      fig, ax1 = plt.subplots()
      ax1.plot(train_acc_list[:epoch], 'r--')
      ax1.plot(test_acc_list[:epoch], 'b-')
      ax1.set_ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.grid(b=True, which='major', color='k', linestyle='-')
      plt.grid(b=True, which='minor', color='k', linestyle='--')
      lgd = plt.legend(['train accuracy', 'test accuracy'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
      ax = plt.gca()
      plt.title('regression costs')
      for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(textsize)
        item.set_weight('normal')
      plt.savefig(graph_dir + '/pred_accuracy.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == "__main__":
  main()

