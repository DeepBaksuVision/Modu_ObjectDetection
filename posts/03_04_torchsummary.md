# Torch summary

이번장에서는 Pytorch에서 모델을 작성할 때, `Keras`에서 제공하는 model summary 처럼 pytorch 모델을 summary해주는 Torch summary module에 대해서 알아보도록 하겠습니다.



이러한 summary 모듈은 해당 네트워크의 구성, 파라미터의 개수, 파라미터의 용량, 연산수을 확인하는데 매우 유용합니다. Torch summary의 원코드는 다음 링크를 참조하시면 됩니다.

- [github : Torch Summary Module](https://github.com/sksq96/pytorch-summary)



이번 섹션에서는 torch summary의 사용법 및 실행했을 때, 어떠한 형태로 출력이 되는지에 대해서 설명드리도록 하겠습니다.



## 01. Installing Torch summary

torch summary 모듈을 설치하는 방법법은 원 코드를 github에서 `clone`하는 방법과 `pip`를 이용하여 설치하는 방법으로 설치할 수 있습니다.



### 01.01) using pip

pip를 이용하여 torch summary를 설치하는 방법은 아래와 같은 명령어를 사용하면 됩니다.



```bash
$ pip install torchsummary

OR

$ pip3 install torchsummary
```



### 01.02) using clone

해당 깃허브의 원코드를 클론받아서 설치하는 방법은 다음과 같습니다.



```bash
$ git clone https://github.com/sksq96/pytorch-summary
```



## 02. How to use torch summary

torch summary를 import하고 사용하는 방법에 대해서 설명하도록 하겠습니다.



### 02.01) import

torch summary 모듈은 다음과 같은 방법으로 import하면 됩니다.



```python3
from torchsummary import summary

# OR

import torchsummary.summary as summary
```



### 02.02) use torch summary

이렇게 import된 torch summary는 정의된 모델에 빈 입력을 넣는 것을 통해서 작동시킬 수 있습니다.



예를들어 torch summary의 `README.md`의 대표적인 예제로 나온 것 처럼 CNN 모델을 작성한다고 가정하면 다음과 같이 모델을 작성할 수 있습니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```



이렇게 작성된 모듈은 다음과 같은 명령어로 객체화 될 수 있고, Device(CPU or GPU)에 할당됩니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)
```



이렇게 까지 진행하고 나면, summary를 통해서 해당 모델의 파라미터를 확인할 수 있습니다.

- summary의 입력 파라미터는 `network model`, `input shape`이 됩니다.



해당 예제의 CNN의 입력 크기는 MNIST를 받으므로 input shape은 `(width, height, channels)`기준으로 `(28, 28, 1)`이 됩니다. pytorch는 input shape을 `(channels, width, height)`으로 받으니 summary의 입력은 `(1, 28, 28)`로 줍니다.



summary 함수를 사용하게 되면 아래와 같은 결과를 확인할 수 있습니다.

해당 결과는 기본적인 네트워크 구성뿐만 아니라 `Number of Parameters` `Input size`, `Forward/backward pass size`, `parameters size`, `Estimated Total Size`에 대한 정보를 출력해줍니다.

```python
summary(model, (1, 28, 28))

>>
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 24, 24]             260
            Conv2d-2             [-1, 20, 8, 8]           5,020
         Dropout2d-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
```

----

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="크리에이티브 커먼즈 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />이 저작물은 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스</a>에 따라 이용할 수 있습니다.

