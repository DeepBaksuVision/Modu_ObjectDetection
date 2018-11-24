# 02) Model

이번 장에서는 YOLO 모델 구조 및 objective function을 설명하고, 해당 내용을 PyTorch 코드레벨로 설명하겠습니다.

앞서 살펴보았듯이 YOLO의 모델 구조는 다음과 같습니다.



![YOLO model](https://user-images.githubusercontent.com/13328380/48620356-3290d280-e9e3-11e8-8488-0d17e57da49b.PNG)

​    

## YOLO class

YOLO의 원 코드인 [Darknet](https://github.com/pjreddie/darknet)의 [yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg) 파일을 확인하는 것을 통해서 YOLO의 pytorch model은 다음과 같이 코드로 구현할 수 있습니다. (완전히 똑같지는 않으나, 유사한 모델설계를 했습니다.)



```python
class YOLOv1(nn.Module):
    def __init__(self, params):

        self.dropout_prop = params["dropout"]
        self.num_classes = params["num_class"]

        super(YOLOv1, self).__init__()
        # LAYER 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 4
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.LeakyReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # LAYER 5
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.LeakyReLU())
        self.layer20 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer21 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer22 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        # LAYER 6
        self.layer23 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())
        self.layer24 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024, momentum=0.01),
            nn.LeakyReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prop)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 7 * 7 * ((10) + self.num_classes))
        )
```

- 초기화할 때, `dropout`, `number of class`에 대한 파라미터를 받는 부분 빼고는 일반적인 YOLO model 형태로 작성하였습니다.

- 중요한 부분은 모델의 마지막 `fc2`의 `7 * 7 * ((10) + self.num_classes)`입니다. 이 부분은 모델의 최종 출력으로 7 x 7 x 30 block tensor이고, 이를 이용하여 Detection에 필요한 요소들을 뽑아서 학습하거나 결과 값을 이용하여 Detection Box를 그리기 때문입니다.

  (여기에서 10은 2개의 box값을 의미합니다. : 1 box (`objectness`, `point_x_shift`, `point_y_shift`, `width_ratio`,`height_ratio`)

​    

## Initialization

초기화는 CNN 파트에서 CNN weights initialization는 `HE initialization`을 해주며, Batch Normalization은 **다음과 같이 초기화해줍니다. **(보강 필요)



```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="leaky_relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

​    

## Forward

YOLO model 설계가 완료되었다면, Forward() 함수를 작성합니다.

마지막에 출력 텐서의 0번째, 10~30까지 `Sigmoid`함수를 준 이유는 0번째 인덱스는 `objectness`를 의미하는 엘리먼트고, 10~30까지의 인덱스는 `class probability`를 의미하기 때문에 해당 값은 확률값을 가져야하기 때문입니다.



``` python
def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)
    out = self.layer9(out)
    out = self.layer10(out)
    out = self.layer11(out)
    out = self.layer12(out)
    out = self.layer13(out)
    out = self.layer14(out)
    out = self.layer15(out)
    out = self.layer16(out)
    out = self.layer17(out)
    out = self.layer18(out)
    out = self.layer19(out)
    out = self.layer20(out)
    out = self.layer21(out)
    out = self.layer22(out)
    out = self.layer23(out)
    out = self.layer24(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc1(out)
    out = self.fc2(out)
    out = out.reshape((-1, 7, 7, ((5) + self.num_classes)))
    out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])
    out[:, :, :, 10:] = torch.sigmoid(out[:, :, :, 10:])

    return out
```



​    

## Objective function

이제 마지막으로 남은 `Objective function`에 대해서 설명하도록 하겠습니다. `Objective function`은 `Cost function` 혹은 `Loss function`이라고 불립니다.



YOLO의 `Objective function`의 수식은 다음과 같습니다.



![Objective function](https://user-images.githubusercontent.com/13328380/48620401-62d87100-e9e3-11e8-8975-93ec5b4eccf8.PNG)



이를 코드로 표현하면 다음과 같습니다.



```python
def detection_loss(output, target):
    from utilities.utils import one_hot

    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape
    _, _, _, n = output.shape

    # calc number of class
    num_of_cls = n - 5

    # class loss
    MSE_criterion = nn.MSELoss()

    # output tensor slice

    # output tensor shape is [batch, 7, 7, 5 + classes]
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    class_output = output[:, :, :, 5:]

    # label tensor slice
    objness_label = target[:, :, :, 0]
    x_offset_label = target[:, :, :, 1]
    y_offset_label = target[:, :, :, 2]
    width_ratio_label = target[:, :, :, 3]
    height_ratio_label = target[:, :, :, 4]
    class_label = one_hot(class_output, target[:, :, :, 5])

    noobjness_label = torch.neg(torch.add(objness_label, -1))

    obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                        (torch.pow(x_offset1_output - x_offset_label, 2) +
                                    torch.pow(y_offset1_output - y_offset_label, 2)))

    obj_size1_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

    objectness_cls_map = torch.stack((objness_label, objness_label, objness_label, objness_label, objness_label), 3)
    objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))
    noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
    obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + noobjness1_loss + objness1_loss + obj_class_loss)
    total_loss = total_loss / b

    return total_loss, obj_coord1_loss / b, obj_size1_loss / b, obj_class_loss / b, noobjness1_loss / b, objness1_loss / b
```



### Objective function 계산 개념

`Objective function`의 계산 개념은 다음과 같은 개념을 사용했습니다.



![loss concept](https://user-images.githubusercontent.com/13328380/48621237-3c680500-e9e6-11e8-9e25-e256192c1648.png)



`objectiveness map`이 나오면, 이를 반전시켜서 `non-objectiveness map`을 생성한 다음에 필요한 조건에 맞춰서 계산한 `Loss`를 계산해 더해줬습니다.



### size loss 이슈

YOLO 논문에 나와있는 그대로 `sqrt`를 모두 씌우게 되면, 네트워크 출력값이 `음수`가 되는순간 `Loss`의 값이 `Nan` 혹은 `-Nan`으로 치닫게 됩니다. 따라서 output tensor의 값은 그대로 둔 상태로 `label`의 값을 `sqrt`하여 학습을 시키는게 일종의 트릭이 됩니다.



```python
obj_size1_loss = lambda_coord * \
                 torch.sum(objness_label *
                 (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                 torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))
```

