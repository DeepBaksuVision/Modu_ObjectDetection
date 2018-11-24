# 01). Dataloader

이번장에서는 Pytorch에서 모델을 작성할 때, 데이터를 feeding 역활을 하는 Dataloder를 작성해보도록 하겠습니다.

해당 챕터의 목록은 다음과 같습니다.

​    

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



일반적으로 이렇게 PyTorch의 Dataset 클래스를 상속받아 커스텀 Dataset 클래스를 만들고 `__getitem__`, `__len__`을 overriding해서 사용합니다.



자세한 내용은 [Pytorch Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)을 확인해 보시기 바랍니다. 

Dataset의 `__getitem__`, `__len__`는 다음과 같은 역활을 합니다.

1.  `__getitem__` :  로드한 data를 차례차례 돌려줌
2. `__len__` : 전체 데이터의 길이를 계산함

​    

추상클래스인 `Dataset`은 최소한  `__getitem__`, `__len__` 함수 구현을 요구합니다. 그럼 이제 Dataset 클래스를 상속받아 Object Detection에 적합한 `VOC`클래스를 만들어봅시다.

​    

## VOC class

VOC class는 앞에 설명한 `convert2Yolo`프로젝트를 같이 이용해서 구현됩니다. VOC class는 추상클래스인 `Dataset`에서 Object Detection data parsing  파일 존재 여부를 확인하는 `_check_exists()`함수에 대한 구현이 추가되며, 추상클래스에서 요구하는 `__getitem__`, `__len__`함수를 overriding하여 내부 함수를 구현합니다.



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

- `root` : 데이터셋의 경로입니다. 일반적으로 PASCAL VOC 데이터의 경우에는 root경로에서 `JPEGImages`, `Annotations`폴더로 분기해서 들어갈 수 있기 때문에, 해당 데이터셋의 폴더 디렉토리 경로가 `root`로 들어가면 편리합니다.

- `train` : 생성될 객체의 용도가 train용인지에 대한 flag 인자입니다.

- `transform` : 추후 설명할 Augmentation 인자입니다. 해당 파라미터는 `입력 데이터`에 대한 변환함수 그 자체입니다.

- `target_transform` : 추후 설명할 Augmentation 인자입니다. 해당 파라미터는 `정답 데이터`에 대한 변환함수 그 자체입니다.

- `resize` : `입력 데이터`에 대한 resize크기입니다.

- `class_path`  : class name이 적혀있는 리스트 파일의 경로입니다.

  (해당 인자의 구조는 `convert2Yolo`와 의존성이 있습니다. 자세한 내용은 [02. convert2Yolo 소개](posts/02_02_Convert2Yolo.md)에서 확인할 수 있습니다.)

​    

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

​    

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



여기서는 [02). convert2Yolo 소개](posts/02_02_Convert2Yolo.md)에서 설명했듯이, `dictionary` 형태의 공통된 파일 포맷에서 yolo format으로 변환 된 `dictionary`형태의 포맷까지 추출하게 됩니다. 

​    

**parsing된 VOC data를 yolo의 label 포맷으로 변경**



yolo format으로 변환된 데이터는 다음과 같은 형태입니다.

```json
{'2008_003829': '1 0.989 0.331 0.022 0.096\n', '2010_004263': '1 0.056 0.544 0.048 0.052\n', '2010_005284': '1 0.887 0.55 0.082 0.12\n4 0.983 0.553 0.03 0.204\n4 0.029 0.517 0.042 0.18\n', '2010_004133': '2 0.61 0.537 0.776 0.659\n', '2009_003238': '1 0.781 0.508 0.438 0.431\n', '2008_004646': '4 0.786 0.595 0.028 0.066\n', ...}
```

위와 같이 `image file name : label`형태의 key-value 쌍으로 구성이 되어있습니다.

- `image file name`을 `key()`함수로 추출한 다음에,  `image file name` 기반으로 sorting을 진행합니다.

- sorting된 `image file name`들의 list파일을 순회합니다.

- `dictionary`형태의 yolo format에서 순회 중인 key이름으로 label 데이터를 가져옵니다.

- 가져온 label 데이터를 개행문자(`\n`)를 기준으로 list 구조로 split합니다.

- split된 string 구조의 label데이터를 학습이 가능하게 float 데이터로 변환해줍니다.

- 이를 `Image file paht : label list structure`형태로 변환한 후, 결과 값을 반환합니다.

  결과값의 구조는 다음과 같습니다.

  ```json
  [{'.../datasets/JPEGImages/2008_008490.jpg': [[0.0, 0.364, 0.267, 0.2, 0.228], [0.0, 0.485, 0.508, 0.246, 0.258]]}, 
  {'.../datasets/JPEGImages/2008_008500.jpg': [[4.0, 0.217, 0.348, 0.062, 0.12]]}, {'.../datasets/JPEGImages/2008_008506.jpg': [[0.0, 0.369, 0.668, 0.41, 0.246]]}, {'.../datasets/JPEGImages/2008_008519.jpg': [[4.0, 0.812, 0.643, 0.032, 0.025]]}, {'.../datasets/JPEGImages/2008_008523.jpg': [[0.0, 0.448, 0.553, 0.892, 0.893]]}, ...]
  ```

​    

### 4. `__len__()`

해당 함수는 위에서  `cvtData()`함수를 이용하여 얻은 학습데이터 리스트의 길이를 확인합니다.



```python3
def __len__(self):
    return len(self.data)
```

​    

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



위에서 언급했듯이, 최종적응로 파싱된 data의 구조는 다음과 같습니다.

```bash
[{'.../datasets/JPEGImages/2008_008490.jpg': [[0.0, 0.364, 0.267, 0.2, 0.228], [0.0, 0.485, 0.508, 0.246, 0.258]]}, 
{'.../datasets/JPEGImages/2008_008500.jpg': [[4.0, 0.217, 0.348, 0.062, 0.12]]}, {'.../datasets/JPEGImages/2008_008506.jpg': [[0.0, 0.369, 0.668, 0.41, 0.246]]}, {'.../datasets/JPEGImages/2008_008519.jpg': [[4.0, 0.812, 0.643, 0.032, 0.025]]}, {'.../datasets/JPEGImages/2008_008523.jpg': [[0.0, 0.448, 0.553, 0.892, 0.893]]}, ...]
```



`__getitem__`함수가 실행되면, 데이터셋의 `index`값이 넘어오게 되는데, 해당 `index`값을 사용하여, `image file path`를 읽어옵니다.

- `list(self.data[index].keys())`를 실행하게 되면, 다음과 같은 결과값을 얻습니다. 

  ```json
  ['.../datasets/JPEGImages/2010_002546.jpg']
  ```

- `image file path`만 가져오기 위해서 `0`번째 인덱스 값을 취하여, `Image`패키지를 이용하여 PIL 이미지를 로드합니다.

- 로드한 이미지의 size를 확인 후에, class 선언시 받은 `resize`파라미터를 이용하여 reszie를 수행합니다.

- label 값을 로드합니다.

- 로드된 이미지와 label값을 인자로 받은 `transform`함수를 이용해 Augmentation을 진행합니다.

- 반환값으론 Augmentation이 완료된 image, target 그리고 원본 이미지의 크기가 반환됩니다. 

​    

### 6. VOC Class

최종 VOC class 코드는 다음과 같습니다.



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

​    

## Dataloader

Object Detection을 위한 커스텀 Dataset을 정의했으니, 이를 이용하여 Dataloader 클래스의 인자로 주면 사용이 가능해집니다.



실제로 train시 Dataset, Dataloader는 다음과 같은 방법으로 사용할 수 있습니다.

```python
train_dataset = VOC(root=data_path, 
                    transform=transforms.ToTensor(), 
                    class_path=class_path)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=detection_collate)
```

- 정의한 VOC class의 인스턴스를 생성합니다.
- 생성된 인스턴스를 `torch.utils.data.DataLoader`의 인자로 줍니다.

​    

### 1. collate_fn

- 이재원님 작성
