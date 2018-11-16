# 01). Dataloader

이번장에서는 Pytorch에서 모델을 작성할 때, 데이터를 feeding 역활을 하는 Dataloder를 작성해보도록 하겠습니다.

해당 챕터의 목록은 다음과 같습니다.



## Pytorch Dataset class

해당 프로젝트에서는 Pytorch Dataset class를 상속받아 data를 parsing하고 있습니다. 따라서 pytorch의 dataset class를 먼저 알아야합니다.



pytorch의 dataset class는 [`torch.utils.data`](https://pytorch.org/docs/stable/data.html)에 있으며, 해당 소스는 [링크](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#Dataset)에서 확인할 수 있습니다.

소스의 원형을 보면 다음과 같습니다.



```python
class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
```



python에서 `__getitem__`, `__len__`, `__add__`에 대해서 이야기하면 더 깊이 이야기할 수 있겠지만, 여기에서는 간단하게 정의하고 지나가겠습니다.

1.  `__getitem__` :  로드한 data를 차례차례 돌려줌
2. `__len__` : 전체 데이터의 길이를 계산함

​    

추상클래스인 `Dataset`은 최소한  `__getitem__`, `__len__` 함수 구현을 요구하고 있으므로, 우리는 이를 Object Detection에 맞춰서 Dataset 클래스를 상속받은 `VOC`클래스를 만들어봅시다.



## VOC class

VOC class는 앞에 설명한 `convert2Yolo`프로젝트를 같이 이용해서 구현됩니다. VOC class는 추상클래스인 `Dataset`에서 Object Detection data parsing  / 파일 존재 여부에 대한 로직이 추가되며, 추상클래스에서 요구하는 `__getitem__`, `__len__`함수를 구현합니다.



### 1. `__init__()`

`__init__()`함수는 VOC class가 초기화될 때, dataloader에서 사용할 수 있도록 최소한의 준비를 하도록 합니다.



`__init__()`함수의 로직흐름은 다음과 같습니다.

- 파라미터 인자를 받아 변수에 할당
- class name list 파일을 받아 load
- Dataset 존재 유무 확인
- Dataset parsing



```python3
IMAGE_FOLDER = "JPEGImages"
LABEL_FOLDER = "Annotations"
IMG_EXTENSIONS = '.jpg'

def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, class_path='./voc.names'):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.resize_factor = resize
    self.class_path = class_path

    with open(class_path) as f:
        self.classes = f.read().splitlines()

    if not self._check_exists():
        raise RuntimeError("Dataset not found.")

    self.data = self.cvtData()
```



먼저 파라미터에 대해서 설명하겠습니다.

- `root` : 데이터셋의 경로입니다.
- `train` : 생성될 객체의 용도가 train용인지에 대한 flag 인자입니다.
- `transform` : 추후 설명할 Augmentation 인자입니다. 해당 파라미터는 `입력 데이터`에 대한 변환함수입니다.
- `target_transform` : 추후 설명할 Augmentation 인자입니다. 해당 파라미터는 `정답 데이터`에 대한 변환함수입니다.
- `resize` : `입력 데이터`에 대한 resize크기입니다.
- `class_path`  : class name이 적혀있는 리스트 파일의 경로입니다.



### 2. `_check_exists()`

해당 함수는 Dataset이 파라미터로 받은 root 경로에 존재하는지 확인합니다.

구현 내용은 다음과 같이 해당 경로의 존재유무를 확인합니다.



```python3
def _check_exists(self):
    print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
    print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

    return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
           os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))
```



### 3. `cvtData()`

해당 함수는 본격적으로 데이터를 파싱하는 함수입니다. 해당 구현은 `convert2Yolo` 프로젝트의 일부를 그대로 사용합니다.



```python3
def cvtData(self):

    result = []
    voc = cvtVOC()

    yolo = cvtYOLO(os.path.abspath(self.class_path))
    flag, self.dict_data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

    try:

        if flag:
            flag, data =yolo.generate(self.dict_data)

            keys = list(data.keys())
            keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

            for key in keys:
                contents = list(filter(None, data[key].split("\n")))
                target = []
                for i in range(len(contents)):
                    tmp = contents[i]
                    tmp = tmp.split(" ")
                    for j in range(len(tmp)):
                        tmp[j] = float(tmp[j])
                    target.append(tmp)

                result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : target})

            return result

    except Exception as e:
        raise RuntimeError("Error : {}".format(e))
```

- VOC data를 먼저 parsing합니다.
- parsing된 VOC data를 yolo의 label 포맷으로 변경해줍니다.
- 변경된 yolo label 포맷을 리스트에 추가합니다.



### 4. `__len__()`

해당 함수는 위에서  `cvtData()`함수를 이용하여 얻은 학습데이터 리스트의 길이를 확인합니다.



```python3
def __len__(self):
    return len(self.data)
```



### 5. `__getitem__()`

해당 함수는 학습 데이터의 일부를 슬라이싱해서 리턴합니다.

Object Detection data를 load한 VOC class의 `__getitem__()`함수는 다음과 같이 구현합니다.



```python3
def __getitem__(self, index):

    key = list(self.data[index].keys())[0]

    img = Image.open(key).convert('RGB')
    current_shape = img.size
    img = img.resize((self.resize_factor, self.resize_factor))

    target = self.data[index][key]

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        # Future works
        pass

    return img, target, current_shape
```



따라서 최종 VOC class 코드는 다음과 같습니다.



```python3
import sys
import os
import torch
import torch.utils.data as data
import numpy as np

from PIL import Image
from convertYolo.Format import YOLO as cvtYOLO
from convertYolo.Format import VOC as cvtVOC

sys.path.insert(0, os.path.dirname(__file__))

class VOC(data.Dataset):

    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=448, class_path='./voc.names'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self.data = self.cvtData()

    def _check_exists(self):
        print("Image Folder : {}".format(os.path.join(self.root, self.IMAGE_FOLDER)))
        print("Label Folder : {}".format(os.path.join(self.root, self.LABEL_FOLDER)))

        return os.path.exists(os.path.join(self.root, self.IMAGE_FOLDER)) and \
               os.path.exists(os.path.join(self.root, self.LABEL_FOLDER))

    def cvtData(self):

        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data =voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data =yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : target})

                return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size
        img = img.resize((self.resize_factor, self.resize_factor))

        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            # Future works
            pass

        return img, target, current_shape
```



