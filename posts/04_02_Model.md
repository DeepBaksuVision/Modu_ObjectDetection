# Model

이번 장에서는 YOLO 모델 구조 및 objective function을 설명하고, 해당 내용을 PyTorch 코드 레벨로 설명하겠습니다.

**실제 코드는 [DeepBaksuVision/You_Only_Look_Once](https://github.com/DeepBaksuVision/You_Only_Look_Once)에서 확인할 수 있습니다.**

앞서 살펴보았듯이 YOLO의 모델 구조는 다음과 같습니다.



![YOLO model](https://user-images.githubusercontent.com/13328380/48620356-3290d280-e9e3-11e8-8488-0d17e57da49b.PNG)

​    

## 01. YOLO class

YOLO의 원 코드인 [Darknet](https://github.com/pjreddie/darknet)의 [yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg) 를 살펴보면서 YOLO의 PyTorch model을 다음과 같이 구현하였습니다 (완전히 동일하지는 않으나, 유사하게 모델을 작성하였습니다).



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
            nn.Linear(4096, 7 * 7 * ((5) + self.num_classes))
        )
```



본 구현체와 You Only Look Once 논문의 내용과 상이한 점은 다음과 같습니다.



**초기화**

- You Only Look Once 논문에서는 ImageNet을 이용하여 classification network를 pre-train 후 4개의 Convolutional Layer를 4개와 2개의 FC를 추가하면서 이를 randomly initialization을 수행합니다.
- 본 구현체는 pre-train 과정을 생략하며, 생략된 pre-train으로 인해 네트워크의 수렴 속도 및 안정성이 떨어진 것을 보강하기 위하여 Convolutional Layer는 HE initialization을 채택하며 FC Layer는 논문에 있는 그대로 randomly initialization을 수행합니다.

​        

**Batch Normalization**

- YOLO 논문에서는 Batch Normalization에 대한 내용은 언급되어있지 않습니다. 그 당시의  [YOLO 모델](https://github.com/pjreddie/darknet/tree/8c5364f58569eaeb5582a4915b36b24fc5570c76/cfg)을 확인해봐도 Batch Normalization을 확인할 수 없습니다(arxiv 기준 batch normalization이 15.02, YOLO가 2015.06입니다. 최근 [Darknet](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)을 확인해보면 기존 YOLO 모델에는 batch normalization이 적용되어있는 것을 확인할 수 있습니다). 
- 본 구현체에서는 논문에는 언급되어있지 않지만, 수렴 속도 및 안정성을 증가시키기 위해 batch normalization을 적용하였습니다.

​    

## 02. Initialization

위에서 언급했듯이 생략된 pre-train으로 인해 네트워크 수렴 속도 및 안정성 하락을 보강하기 위하여 `He initialization`을 수행합니다. `He initialization`은 Pytorch에서 제공하는 `torch.nn.init.kaiming_normal_`함수를 이용합니다.

Batch Normalization은 $$ \gamma $$ 값을 `1`, $$ \beta $$값을 `0`으로 초기화합니다. 이는 $$ \gamma $$가 scale, $$ \beta $$가 shift 값을 의미하기 때문에 초기의 layer 출력값에서 batch normalization layer의해 scale 및 shift가 안일어나는 값으로 설정했습니다.



```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="leaky_relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

​    

## 03. Forward

YOLO model 설계가 완료되었다면, Forward() 함수를 작성합니다.

마지막에 출력 텐서의 0번째, 10~30까지 `Sigmoid`함수를 준 이유는 0번째 인덱스는 `objectness`를 의미하는 엘리먼트고, 5~25까지의 인덱스는 `class probability`를 의미하기 때문에 해당 값은 확률값을 가져야 하기 때문입니다.



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
    out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])  # sigmoid to objness1_output
    out[:, :, :, 5:] = torch.sigmoid(out[:, :, :, 5:])  # sigmoid to class_output

    return out
```



​    

## 04. Objective function

이제 마지막으로 남은 `Objective function`에 대해서 설명하도록 하겠습니다. `Objective function`은 `Cost function` 혹은 `Loss function`이라고 불립니다.



YOLO의 `Objective function`의 수식은 다음과 같습니다.



![Objective function](https://user-images.githubusercontent.com/13328380/48620401-62d87100-e9e3-11e8-8975-93ec5b4eccf8.PNG)



이를 코드로 표현하면 다음과 같습니다.



```python
def detection_loss_4_yolo(output, target, device):
    from utilities.utils import one_hot

    # hyper parameter

    lambda_coord = 5
    lambda_noobj = 0.5

    # check batch size
    b, _, _, _ = target.shape
    _, _, _, n = output.shape

    # output tensor slice
    # output tensor shape is [batch, 7, 7, 5 + classes]
    objness1_output = output[:, :, :, 0]
    x_offset1_output = output[:, :, :, 1]
    y_offset1_output = output[:, :, :, 2]
    width_ratio1_output = output[:, :, :, 3]
    height_ratio1_output = output[:, :, :, 4]
    class_output = output[:, :, :, 5:]

    num_cls = class_output.shape[-1]

    # label tensor slice
    objness_label = target[:, :, :, 0]
    x_offset_label = target[:, :, :, 1]
    y_offset_label = target[:, :, :, 2]
    width_ratio_label = target[:, :, :, 3]
    height_ratio_label = target[:, :, :, 4]
    class_label = one_hot(class_output, target[:, :, :, 5], device)

    noobjness_label = torch.neg(torch.add(objness_label, -1))

    obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                        (torch.pow(x_offset1_output - x_offset_label, 2) +
                                    torch.pow(y_offset1_output - y_offset_label, 2)))

    obj_size1_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

    objectness_cls_map = objness_label.unsqueeze(-1)

    for i in range(num_cls - 1):
        objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

    obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

    noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
    objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))

    total_loss = (obj_coord1_loss + obj_size1_loss + noobjness1_loss + objness1_loss + obj_class_loss)
    total_loss = total_loss / b

    return total_loss, obj_coord1_loss / b, obj_size1_loss / b, obj_class_loss / b, noobjness1_loss / b, objness1_loss / b
```



### Objective function 계산 개념



앞서 dataloader를 설명했던 챕터에서 collate_fn 함수에 대한 설명을 진행하면서 모델의 output tensor에 대해서 언급했었습니다.



다시 한번 YOLO의 output tensor를 확인해보겠습니다.

![YOLO output tensor](https://user-images.githubusercontent.com/15168540/48966993-9a679e80-f01d-11e8-8f78-66a7135859eb.png)

- `[x, y, w, h]` : box의 좌표값
  - `x` : 각 grid cell의 좌상단`(0, 0)`에서 x축 방향 offset 값
  - `y` : 각 grid ceell의 좌상단 `(0, 0)`에서 y축 방향 offset 값
  - `w`: 전체 Image의 width와 박스 width의 비율(ratio)값
  - `h`: 전체 Image의 height와 박스 height의 비율(ratio)값
- `c` : `objectness` 확률, 해당 grid cell에 Object가 존재하는지 하지 않는지에 대한 확률값
- `class probabilities` : 해당 grid cell에서 각 class의 확률값



Cost value는 YOLO model에서 얻은 output tensor값을 이용하여 계산하게 됩니다.

Objective function에서 사용되는 몇 가지 파라미터들은 다음과 같이 설정했습니다.

- `lambda coord` : 5

- `lambda noobj` : 0.5

- $$ 1^{obj}_{ij} $$: label의 object 센터값에 해당하는 grid 위치는 `1`, 그렇지 않은 위치는 `0`으로 맵핑하여 object map을 생성

  ```python
  objness_label = target[:, :, :, 0]
  ```

- $$ 1^{noobj}_{ij} $$ :  $$1^{obj}_{ij}$$의 반대개념이므로, $$ 1^{obj}_{ij} $$를 inverse해서 non-object map을 생성

  ```python
  noobjness_label = torch.neg(torch.add(objness_label, -1))    
  ```

​    

이제 본격적으로 cost value를 계산해보도록 하겠습니다. 기본적인 개념은 python의 slicing을 이용하여 `[w, h, c]` 형태로 단면들을 자르고 이를 tensor block의 operation으로 연산하는 것입니다.



**slicing**

```python
# output tensor slice
# output tensor shape is [batch, 7, 7, 5 + classes]
objness1_output = output[:, :, :, 0]
x_offset1_output = output[:, :, :, 1]
y_offset1_output = output[:, :, :, 2]
width_ratio1_output = output[:, :, :, 3]
height_ratio1_output = output[:, :, :, 4]
class_output = output[:, :, :, 5:]

num_cls = class_output.shape[-1]

# label tensor slice
objness_label = target[:, :, :, 0]
x_offset_label = target[:, :, :, 1]
y_offset_label = target[:, :, :, 2]
width_ratio_label = target[:, :, :, 3]
height_ratio_label = target[:, :, :, 4]
class_label = one_hot(class_output, target[:, :, :, 5], device)
```



slicing이 완료됬다면 이를 이용하여 cost를 구합니다.

**multi-task cost**

```python
obj_coord1_loss = lambda_coord * \
                      torch.sum(objness_label *
                        (torch.pow(x_offset1_output - x_offset_label, 2) +
                                    torch.pow(y_offset1_output - y_offset_label, 2)))

obj_size1_loss = lambda_coord * \
                     torch.sum(objness_label *
                               (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                                torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))

objectness_cls_map = objness_label.unsqueeze(-1)

for i in range(num_cls - 1):
    objectness_cls_map = torch.cat((objectness_cls_map, objness_label.unsqueeze(-1)), 3)

obj_class_loss = torch.sum(objectness_cls_map * torch.pow(class_output - class_label, 2))

noobjness1_loss = lambda_noobj * torch.sum(noobjness_label * torch.pow(objness1_output - objness_label, 2))
objness1_loss = torch.sum(objness_label * torch.pow(objness1_output - objness_label, 2))
```

- `objectness_cls_map`을 stack을 class 개수 만큼 해주는 이유는 label이 onehot encoding을 거쳐 각 요소별로 class cost를 계산하기 때문입니다. (`class map` 생성)



이렇게 각 Multi-task에 대한 cost를 개별적으로 구했다면, 이를 합하여 `total_loss`로 합치고, 이를 반환해주는 함수를 작성합니다.

​    

### size loss 이슈



코드를 유심히 본 독자들은 눈치챘을 수도 있겠습니다. YOLO 논문의 Objective function에는 output tensor와 label tensor에 `sqrt`를 적용했지만, 코드상에는 label tensor에만 `sqrt`를 놓고 output tensor 값은 그대로 사용합니다.



팀 프로젝트를 진행하면서 YOLO를 학습하려고 했을 때 Objective function을 그대로 적용했으나 weights initialization 및 초기 학습 시 네트워크가 불안정하면서 output tensor값이 음수가 발생하는 경우가 있습니다. 이때, Cost가 `Nan`이 뜨고 되고 학습이 안 되는 현상을 발견하였습니다.



해당 사항에 대해서 팀원끼리 논의 및 원 코드를 확인한 결과 저자 코드에서 위와 같이 label에만 `sqrt`값을 사용하고 output tensor에는 `sqrt`를 적용하지 않는 것을 확인하고 원저자 코드의 흐름을 따르게 되었습니다.



```python
obj_size1_loss = lambda_coord * \
                 torch.sum(objness_label *
                 (torch.pow(width_ratio1_output - torch.sqrt(width_ratio_label), 2) +
                 torch.pow(height_ratio1_output - torch.sqrt(height_ratio_label), 2)))
```

---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="크리에이티브 커먼즈 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />이 저작물은 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스</a>에 따라 이용할 수 있습니다.

