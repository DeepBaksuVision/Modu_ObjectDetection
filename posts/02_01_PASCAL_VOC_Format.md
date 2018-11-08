# 01). PASCAL VOC Dataset



- PASCAL VOC Dataset에 대한 설명 진행



## 폴더 계층 구조

PASCAL VOC Dataset을 다운받아 압축을 풀게되면 다음과 같은 구조를 확인할 수 있습니다.

​    

```bash
VOC20XX
├── Annotations
├── ImageSets
├── JPEGImages
├── SegmentationClass
└── SegmentationObject
```



- `Annotations` : `JPEGImages` 폴더 속 원본 이미지와 같은 이름들의 `xml`파일들이 존재합니다. Object Detection을 위한 `label`이 됩니다.
- `ImageSets` : 어떤 이미지 그룹을 `test`, `train`, `trainval`, `val`로 사용할 것인지, 특정 클래스가 어떤 이미지에 있는지 등에 대한 정보들을 포함하고 있는 폴더입니다. 
- `JPEGImages` : `*.jpg`확장자를 가진 이미지 파일들이 모여있는 폴더입니다. Object Detection에서 `data`가 됩니다.
- `SegmentationClass` :  Semantic segmentation을 학습하기 위한 `label` 이미지입니다.
- `SegmentationObject` : Instance segmentation을 학습하기 위한 `label` 이미지입니다.

​    

Object Detection을 할때는 주로 `Annotations`, `JPEGImages`폴더가 사용됩니다. `data`로 사용되는 이미지의 경우에는 그냥 load해서 사용하면 되나, `label`의 경우는 parsing이 필요한 경우가 있으므로 `Annotations`의 `*.xml` 구조에 대해서 살펴보도록 하겠습니다.

​    

### xml 파일 구조

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

### Python에서 xml파일 load하고 값 확인하기

Object Detection에서 `label`을 파싱하기 위해서 python의 xml package를 이용하여 xml파일을 load하는 예제를 진행해보겠습니다.

​    

PASCAL VOC의 `Annotations`폴더의 xml파일을 load하고 값을 추출하는 `load.py`스크립트는 아래와 같습니다.

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

- load한 xml파일의 값들을 이용해서 Image에 box 그리기