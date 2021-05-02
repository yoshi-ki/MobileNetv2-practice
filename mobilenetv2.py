import torch.nn as nn
import math


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

def conv_3x3_bn(ipc, opc, stride):
  # function used for the first layer in mobilenetv2
  return nn.Sequential(
    nn.Conv2d(ipc, opc, 3, stride, 1, bias=False),
    nn.BatchNorm2d(opc),
    nn.ReLU6(inplace=True)
  )

def conv_1x1_bn(ipc, opc):
  return nn.Sequential(
    nn.Conv2d(ipc, opc, 1, 1, 0, bias=False),
    nn.BatchNorm2d(opc),
    nn.ReLU6(inplace=True)
  )

class Inverted_Residual(nn.Module):
  def __init__(self, ipc, opc, stride, expand_ratio):
    super(Inverted_Residual, self).__init__()
    hidden_dim = int(ipc * expand_ratio)
    self.res_connect = ((stride == 1) and (ipc == opc))

    if (expand_ratio == 1):
      self.bottleneck = nn.Sequential(
        # 1st
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # 2nd
        nn.Conv2d(hidden_dim, opc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(opc)
      )
    else:
      self.bottleneck = nn.Sequential(
        # 1st
        nn.Conv2d(ipc, hidden_dim, 1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # 2nd
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU6(inplace=True),
        # 3rd
        nn.Conv2d(hidden_dim, opc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(opc)
      )

  def forward(self,x):
    if self.res_connect:
      return x + self.bottleneck(x)
    else:
      return self.bottleneck(x)



class MobileNetv2(nn.Module):
  def __init__(self, n_class=10, mult_width_param=1.):
    super(MobileNetv2, self).__init__()
    self.config = [
      # t(expansion factor), c(output channel), n(iteration), s(stride)
      [1, 16, 1, 1],
      [6, 24, 2, 2],
      [6, 32, 3, 2],
      [6, 64, 4, 2],
      [6, 96, 3, 1],
      [6, 160, 3, 2],
      [6, 320, 1, 1],
    ]

    # build the first layer
    ipc = 32
    self.features = [conv_3x3_bn(3, ipc, 2)]

    # build bottleneck blocks
    block = Inverted_Residual
    for t, c, n, s in self.config:
      opc = _make_divisible(c * mult_width_param, 4 if mult_width_param <= 0.1 else 8)
      for i in range(n):
        self.features.append(block(ipc, opc, s if i == 0 else 1, expand_ratio=t))
        ipc = opc

    # build last several layers
    last_channel = _make_divisible(1280 * mult_width_param, 4 if mult_width_param == 0.1 else 8) if mult_width_param > 1.0 else 1280
    self.features.append(conv_1x1_bn(ipc, last_channel))
    self.features = nn.Sequential(*self.features)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(last_channel, n_class)

    self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


if __name__ == '__main__':
  net = MobileNetv2()