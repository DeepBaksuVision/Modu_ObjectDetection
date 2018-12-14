# PASCAL VOC Dataset

본 챕터는 Object Detection dataset 중 하나인 PASCAL VOC dataset의 구조와 이를 이용하여 label 데이터와 이미지가 어떻게 그려지는지 설명합니다.



이를 이용하면 Object Detection에서 사용 가능한 dataloader를 만들 수 있게됩니다.



## 01. Object Detection Label 이미지에 시각화하기

해당 챕터에서는 Object Detection label을 이미지에 시각화하여 실제로 Object detection label이 어떻게 구성되어있는지 잘펴보도록 하겠습니다.

​    

해당 챕터의 구성은 다음과 같습니다.

1. **PASCAL VOC 폴더 계층 구조**
2. **Image 시각화**
3. **XML 파일 구조**
4. **XML파일 값 로드**
5. **Object Detection label 시각화**

​    

## 02. PASCAL VOC 폴더 계층 구조

PASCAL VOC Dataset을 다운받아 압축을 풀게되면 다음과 같은 구조를 확인할 수 있습니다.

```bash
VOC20XX
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
```

- `Annotations` : `JPEGImages` 폴더 속 원본 이미지와 같은 이름들의 `xml`파일들이 존재합니다. Object Detection을 위한 `정답 데이터`이 됩니다.
- `ImageSets` : 어떤 이미지 그룹을 `test`, `train`, `trainval`, `val`로 사용할 것인지, 특정 클래스가 어떤 이미지에 있는지 등에 대한 정보들을 포함하고 있는 폴더입니다.
- `JPEGImages` : `*.jpg`확장자를 가진 이미지 파일들이 모여있는 폴더입니다. Object Detection에서 `입력 데이터`가 됩니다.
- `SegmentationClass` :  Semantic segmentation을 학습하기 위한 `label` 이미지입니다.
- `SegmentationObject` : Instance segmentation을 학습하기 위한 `label` 이미지입니다.

​    

Object Detection을 할때는 주로 `Annotations`, `JPEGImages`폴더가 사용됩니다. 모델에 입력으로 넣는 `입력데이터`인 경우 그냥 load해서 사용하면 되나,  지도학습에 핵심이 되는 `정답 데이터`의 경우는 parsing이 필요한 경우가 있으므로 `Annotations`의 `*.xml` 구조는 잘 알아두는 것이 중요합니다.

​    

## 03. Image 시각화

이제부터 본격적으로 Object Detection을 위한 데이터셋을 시각화하는 방법에 대해서 설명하도록 하겠습니다.

먼저 `입력 데이터`인 이미지 파일을 python에서 로드하고, 이를 시각화하는 방법에 대해서 소개하도록 하겠습니다.

​    

스크립트 코드는 다음과 같습니다.

**load_image.py**

```python
import os
import sys
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

image_path = sys.argv[1]

image = Image.open(image_path).convert("RGB")

plt.figure(figsize=(25,20))
plt.imshow(image)
plt.show()
plt.close()
```

- 파라미터로 이미지의 `경로 + 파일 이름`을 받습니다.
- `PIL` 패키지를 이용하여 이미지를 load합니다.
- `matplotlib.pyplot` 패키지를 이용하여 이미지를 시각화합니다.



스크립트는 다음과 같은 명령어로 실행할 수 있습니다.

```bash
$ python3 load_image.py <image file path>

>> python3 load_image.py ./VOCdevkit/JPEGImages/2007_000068.jpg
```



해당 스크립트를 실행하면 다음과 같은 결과를 확인할 수 있습니다.

![result of load_image.py](https://user-images.githubusercontent.com/13328380/48310553-5499e780-e5d4-11e8-8f8d-70a93b4cf2ae.png)



## 04. XML 파일 구조

xml 파일 안에는 수많은 tag들이 존재하지만 Object Detection 모델을 학습하기 위해 사용되는 tag들은 정해져있습니다. 따라서 해당 섹션에서는 필요한 tag들이 어떤 의미를 갖는지 설명하고 해당 xml을 읽어들여서 해당 tag의 값을 가져오는 python 예제 코드에 대해서 설명하도록 하겠습니다.

​    

PASCAL VOC Dataset의 `Annotations`에 있는 xml파일들의 구조는 다음과 같습니다.



```xml
<annotation>
	<folder>VOC2007</folder>
	<filename>000001.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<owner>
		<flickrid>Fried Camels</flickrid>
		<name>Jinky the Fruit Bat</name>
	</owner>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>
```



- `<size>` : xml파일과 대응되는 이미지의 width, height, channels 정보에 대한 tag입니다.
  - `<width>` : xml파일에 대응되는 이미지의 width값입니다.
  - `<height>` : xml파일에 대응되는 이미지의 height값입니다.
  - `<depth>` : xml파일에 대응되는 이미지의 channels값입니다.
- `<object>` : xml파일과 대응되는 이미지속에 object의 정보에 대한 tag입니다.
  - `<name>` : 클래스 이름을 의미합니다.
  - `<bndbox>` : 해당 object의 바운딩상자의 정보에 대한 tag입니다.
    - `xmin` : object 바운딩상자의 왼쪽상단의 x축 좌표값입니다.
    - `ymin` : object 바운딩상자의 왼쪽상단의 y축 좌표값입니다.
    - `xmax` : object 바운딩상자의 우측하단의 x축 좌표값입니다.
    - `ymax` : object 바운딩상자의 우측하단의 y축 좌표값입니다.

​    

바운딩 박스에 대한 좌표값을 시각적으로 표현하면 다음과 같습니다.

![Bounding_box_visualization](https://user-images.githubusercontent.com/13328380/48208872-23909b80-e3b7-11e8-9cf6-c9d502015823.png)

​    

## 05. XML파일 load

Object Detection에서 `label`을 파싱하기 위해서 python의 xml package를 이용하여 xml파일을 load하는 예제를 진행해보겠습니다.

​    

**load.py**

```python
import sys
import os
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

xml_path = sys.argv[1]

print("XML parsing Start\n")
xml = open(xml_path, "r")
tree = Et.parse(xml)
root = tree.getroot()

size = root.find("size")

width = size.find("width").text
height = size.find("height").text
channels = size.find("depth").text

print("Image properties\nwidth : {}\nheight : {}\nchannels : {}\n".format(width, height, channels))

objects = root.findall("object")
print("Objects Description")
for _object in objects:
    name = _object.find("name").text
    bndbox = _object.find("bndbox")
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    xmax = bndbox.find("xmax").text
    ymax = bndbox.find("ymax").text

    print("class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n".format(name, xmin, ymin, xmax, ymax))

print("XML parsing END")
```



위의 코드의 로직 흐름은 다음과 같습니다.

- 입력된 xml 파일 위치를 받아 xml파일을 open합니다.
- open한 xml파일을 이용하여 `xml.etree.ElementTree` 를 사용하여 파싱합니다.
- `xml.etree.ElementTree`객체로부터 xml의 루트(여기에서는 <annotation> tag)를 가져옵니다.
- `<size>` tag를 찾습니다.
- 찾은 `<size>` tag에서 `width`, `height`, `depth` tag를 찾아서 값들을 불러옵니다.
- xml의 루트에서 모든 `<objects>` tag를 찾습니다.
- 여러 `<objects>` tags을 순회하면서 `<name>`, `<bndbox>`, `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>` tag를 찾아서 값들을 불러옵니다.

​    

해당 코드는 다음과 같은 명령어를 통해서 실행이 가능합니다.

(명령어는 linux기반으로 작성되었습니다.)



```bash
$ python3 load.py <xml file path>

ex)
$ python3 load.py /home/ubuntu/VOCdevkit/VOC2007/Annotations/000001.xml

>>
XML parsing Start

Image properties
width : 353
height : 500
channels : 3

Objects Description
class : dog
xmin : 48
ymin : 240
xmax : 195
ymax : 371

class : person
xmin : 8
ymin : 12
xmax : 352
ymax : 498

XML parsing END
```

​        

## 06. Object Detection label 시각화

이제 위에서 얻은 label 정보를 이용하여 이미지에 Object Detection box를 그려보도록 하겠습니다.

Object Detection label 정보를 이미지에 그리는 로직 흐름은 다음과 같습니다.



- `Annotations`폴더의 파일 리스트를 불러옴
- `JPEGImages`폴더에서 XML 파일과 대응되는 이미지를 찾음
- 파일 리스트를 순회를 돌면서 파일에서 label 정보를 load함
- load된 label 정보를 이용하여 Object Detection 정보를 시각화함



이러한 로직 흐름을 위에서 구현한 `load.py`와 몇가지 스크립트 파일을 이용하여 점진적으로 구현해보도록 하겠습니다.



### 06.01) os.walk 를 이용해 디렉토리의 파일 리스트를 가져오기

os package의 walk함수를 사용하면, 특정 디렉토리의 `경로`, `폴더 리스트`, `파일 리스트`를 가지고 올 수 있습니다.

​    

**directory_search.py**

```python3
import os
import sys

dataset_path = sys.argv[1]

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

print("ROOT : {}\n".format(ann_root))
print("DIR : {}\n".format(ann_dir))
print("FILES : {}\n".format(ann_files))
```

위의 스크립트는 PASCAL VOC 데이터셋 폴더를 기준으로 작성되었습니다.

- 파라미터로 PASCAL VOC Object Detection dataset 경로를 받습니다.

- `os.walk` 함수를 이용하여 `Annotations`폴더의 `경로`, `폴더 리스트`, `파일 리스트`정보를 담은 iterator를 반환합니다.

-  `next`함수를 이용하여 iterator의 첫번째 값을 가지고 옵니다.

  (`os.walk` iterator의 첫번째 값은 3개의 값을 리턴하는데, 차례대로 `경로`, `폴더 리스트`, `파일 리스트`입니다.)

- 얻은 3가지의 값을 출력합니다.

​    

해당 스크립트 파일은 다음과 같은 명령어로 실행할 수 있습니다.

``` bash
$ python3 directory_search.py <directory>

>> python3 directory_search.py ./VOCdevkit/
ROOT : ./Annotations

DIR : []

FILES : ['2008_003829.xml', '2010_004263.xml', '2010_005284.xml', '2010_004133.xml', '2009_003238.xml', '2008_004646.xml', '2007_004112.xml', '2008_001482.xml', '2008_002043.xml', '2010_005498.xml', ... ]
```



### 06.02) xml파일을 load하기

얻은 `파일리스트`를 순회하며 위에서 작성했던 스크립트 파일(`load.py`)를 이용하여 xml 파일을 로드합니다.

​    

**searchNload.py**

```python3
import os
import sys
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

dataset_path = sys.argv[1]

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

for xml_file in ann_files:
    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    print("Image properties\nwidth : {}\nheight : {}\nchannels : {}\n".format(width, height, channels))

    objects = root.findall("object")
    print("Objects Description")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = bndbox.find("xmin").text
        ymin = bndbox.find("ymin").text
        xmax = bndbox.find("xmax").text
        ymax = bndbox.find("ymax").text

        print("class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n".format(name, xmin, ymin, xmax, ymax))

    print("XML parsing END")

```

- `os.walk`를 이용해 얻은 `파일 리스트`를 순회하면서, xml파일을 load하여 얻은 값들을 출력합니다.

​    

해당 스크립트 파일은 다음과 같은 명령어로 실행할 수 있습니다.

```bash
$ python3 searchNload.py <directory>

>> python3 searchNload.py ./VOCdevkit/

Image properties
width : 500
height : 375
channels : 3

Objects Description
class : car
xmin : 489
ymin : 106
xmax : 500
ymax : 142

XML parsing END
Image properties
width : 500
height : 286
channels : 3

Objects Description
class : car
xmin : 16
ymin : 148
xmax : 40
ymax : 163

...
```



### 06.03) 각 xml파일에서 얻은 값들을 이용하여 원 Image에 box정보 그리기

지금까지 특정 디렉토리에 있는 파일 리스트를 가져와서 xml파일 정보를 가져오는 스크립트를 구현하였습니다.

이제 마지막으로 같은 방식으로 이미지 폴더에서 xml파일과 매칭되는 이미지를 불러들여와서 해당 이미지에 xml파일에서 얻은 정보를 시각화하는 스크립트 파일을 구현해보도록 하겠습니다.

​    

**draw_detection_box.py**

```python3
import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

dataset_path = sys.argv[1]

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

for xml_file in ann_files:

    # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
    img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]
    img_file = os.path.join(img_root, img_name)
    image = Image.open(img_file).convert("RGB")
    draw = ImageDraw.Draw(image)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)

    plt.figure(figsize=(25,20))
    plt.imshow(image)
    plt.show()
    plt.close()
```

- `1. os.walk를 이용해 디렉토리 파일 리스트를 가져오기`와 같은 방법으로 Image폴더의 파일 리스트를 가져옵니다.

- xml 파일 리스트를 순회하면서 xml파일과 일치하는 이미지 파일 경로를 찾습니다.
- xml파일로부터 Object Detection label정보를 얻습니다.
- 이미지 파일을 로드합니다.
- 로드한 이미지 파일에 획득한 Object Detection label정보를 시각화합니다.

​    

해당 스크립트 파일은 다음과 같은 명령어로 실행할 수 있습니다.

```bash
$ python3 searchNload.py <directory>

>> python3 searchNload.py ./VOCdevkit/
```

​    

실행하게 되면, 다음과 같은 화면을 볼 수 있습니다.

![object detection label visualization](https://user-images.githubusercontent.com/13328380/48310215-071a7c00-e5ce-11e8-855e-c60651251f69.png)



---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="크리에이티브 커먼즈 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />이 저작물은 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스</a>에 따라 이용할 수 있습니다.