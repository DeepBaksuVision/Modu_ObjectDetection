# 02). convert2Yolo 소개



[convert2Yolo](https://github.com/ssaru/convert2Yolo)는 각종 datasets들을 YOLO[You Only Look Once]저자가 만든 [darknet 프레임워크](https://pjreddie.com/darknet/)가 사용하는 label format으로 변경해주는 프로젝트입니다.



본 문서에서 모든 Object Detection 구현체의 Dataloader는 [convert2Yolo](https://github.com/ssaru/convert2Yolo)를 이용하므로 해당 프로젝트가 어떻게 구성되어있고 어떻게 사용하는지 숙지해야 Dataloader를 이해할 수 있습니다.

​    

## 02-01).사용법

[convert2Yolo](https://github.com/ssaru/convert2Yolo)의 README에 사용법이 간단히 정리되어있습니다. 

해당 내용을 정리해보겠습니다.

​    

### 지원하는 데이터셋

[convert2Yolo](https://github.com/ssaru/convert2Yolo)는 현재 아래와 같은 데이터셋을 지원합니다. 

- COCO
- VOC
- UDACITY
- KITTI



해당 데이터셋이 아닌 다른 데이터셋을 사용하려면 직접 파싱코드를 작성하는 방법, [convert2Yolo](https://github.com/ssaru/convert2Yolo)에 해당 데이터셋을 파싱하는 코드를 작성, 새로운 툴을 사용하는 방법으로 이를 해결해야합니다.

​    

### 의존성 패키지 설치

먼저 [convert2Yolo](https://github.com/ssaru/convert2Yolo)는 여러 의존성 패키지 툴을 필요로합니다. 

해당 코드를 클론한 다음 다음과 같은 명령어를 이용하여 의존성 패키지를 설치해줍니다.



```bash
$ mkdir convert2Yolo
$ cd convert2Yolo 
$ git init 
$ git remote add origin https://github.com/ssaru/convert2Yolo.git
$ git pull origin master
$ pip3 install -r requirements.txt
```

​    

### 파라미터

[convert2Yolo](https://github.com/ssaru/convert2Yolo)는 여러가지 파라미터를 받습니다. 

파라미터 종류는 아래와 같습니다.

- `--datasets`
- `--img_path`
- `--label`
- `--convert_output_path`
- `--img_type`
- `--manipast_path`
- `--cla_list_file`

​    

**`--datasets`** : 해당 파라미터는 어떤 dataset에 대해서 데이터셋 파싱을 진행할건지에 대한 mode flag입니다. 위에서도 언급했듯이 [convert2Yolo](https://github.com/ssaru/convert2Yolo)는 `COCO`, `VOC`, `UDACITY`, `KITTI` 데이터셋을 지원하므로 해당 파라미터도  `COCO`, `VOC`, `UDACITY`, `KITTI`중에 하나를 입력하면 됩니다.

```bash
$ --datasets VOC
```

​    

**`--img_path`** : 데이터셋의 이미지 폴더 경로입니다. 예를들어 데이터셋이 `PASCAL VOC`라고 가정했을 때, 이미지 폴더는 `VOC20XX/JPEGImages`에 있을테니 다음과 같이 파라미터를 입력하면 됩니다.

```bash
$ --img_path VOC20XX/JPEGImages/
```

​    

**`--label`** : 데이터셋의 레이블 폴더 경로입니다. 예를들어 데이터셋이 `PASCAL VOC`라고 가정했을 때, 레이블 폴더는 `VOC20XX/Annotations`에 있을테니 다음과 같이 파라미터를 입력하면 됩니다.

```bash
$ --label VOC20XX/Annotations/
```



어떤 데이터셋들은 레이블이 복수의 파일이 아닌 `*.json`, `*.csv` 확장자를 갖는 단일파일인 경우가 있습니다. 대표적으로 `COCO`, `UDACITY`같은 경우가 대표적인데, 이런 경우에는 `--label`파라미터는 다음과 같이 파일 경로를 입력해주면 됩니다.

```bash
$ --label ./UDACITY/label.csv
```

​    

**`--convert_output_path`** : 변환된 레이블파일이 저장될 폴더 경로를 의미합니다. 사용자가 원하는 폴더를 생성하거나 이미 있는 폴더를 지정해주면 됩니다.

```bash
$ --convert_output_path ./
```

​    

**`--img_type`** : `--img_path` 폴더에 있는 이미지의 확장자가 무엇인지 명시해주는 파라미터입니다. [convert2Yolo](https://github.com/ssaru/convert2Yolo)는 `*.jpg`, `*.png`파일을 지원합니다.

```bash
$ --img_type ".jpg"
or
$ --img_type ".png"
```

​    

**`--manipast_path`** : [darknet 프레임워크](https://pjreddie.com/darknet/)을 이용하여 학습할 경우에는 데이터셋의 이미지가 어디있는지 각 이미지마다 파일 경로가 적혀있는 `*.txt`파일을 요구하게됩니다. 해당 파라미터는 [darknet 프레임워크](https://pjreddie.com/darknet/)를 위한 파라미터이지만, 불필요한 경우에 생략할 수 없으므로 무조건 적어주어야 합니다.

```bash
$ --minipast_path ./
```

​     

**`--cls_list_file`** :  해당 파라미터 또한 [darknet 프레임워크](https://pjreddie.com/darknet/)에서 필요한 파일입니다. [darknet 프레임워크](https://pjreddie.com/darknet/)에서는 클래스 목록을 `*.names`파일을 참조하여 학습하고, 추론하게됩니다. 해당 파일의 포맷은 [링크](https://github.com/pjreddie/darknet/blob/master/data/voc.names)를 참조하시면 됩니다.

```bash
$ --cls_list_file voc.names
```

​    

### 예제 코드

파라미터에 대한 내용을 숙지했다면, 다음과 같은 명령어로 특정 dataset을 [darknet 프레임워크](https://pjreddie.com/darknet/)에서 필요로하는 레이블 포맷으로 변환할 수 있습니다.



```bash
$ python3 example.py --datasets [COCO/VOC/KITTI/UDACITY] --img_path <image_path> --label <label path or annotation file> --convert_output_path <output path> --img_type [".jpg" / ".png"] --manipast_path <output manipast file path> --cls_list_file <*.names file path>

>> python3 example.py --datasets KITTI --img_path ./example/kitti/images/ --label ./example/kitti/labels/ --convert_output_path ./ --img_type ".jpg" --manipast_path ./ --cls_list_file names.txt
```

​    

## 02-02) 변환 방식

[convert2Yolo](https://github.com/ssaru/convert2Yolo)의 변환방식은 기본적으로 다음과 같은 방식을 따릅니다.



1. 데이터 Load (`parsing`)
2. `dictionary`형태의 공통된 label format으로 변경 (`generate`)
3. yolo format으로 변경 (`generate`)
4. 저장 (`save`)



`dictionary`형태의 공통된 파일 포맷은 다음과 같습니다.

```json
{
    "filename" :      
                {                  
                    "size" :
                                {
                                    "width" : <string>
                                    "height" : <string>
                                    "depth" : <string>
                                }
                
                    "objects" :
                                {
                                    "num_obj" : <int>
                                    "<index>" :
                                                {
                                                    "name" : <string>
                                                    "bndbox" :
                                                                {
                                                                    "xmin" : <float>
                                                                    "ymin" : <float>
                                                                    "xmax" : <float>
                                                                    "ymax" : <float>
                                                                }
                                                }
                                    ...
                
                
                                }
                }
...
}
```

- `filename`은 확장자가 제거된 image file name입니다.
- `size`는 image의 해상도입니다.
- `objects`는 해당 이미지에 있는 객체 전체를 포함합니다.
- `num_obj`는 `objects`의 개수입니다.
- `<index>`는 `objects`에서 특정 object 일부를 표현합니다.
- `bndbox`는 object의 box 좌표정보에 대한 정보입니다.

​    

이러한 변환방식을 이해했다면, `VOC` 데이터셋을 `YOLO`포맷으로 바꾸는 과정에 대한 다음 코드를 통해 더 자세하게 [convert2Yolo](https://github.com/ssaru/convert2Yolo)가 어떻게 작동하는지 확인해봅시다.



```python
voc = VOC()
yolo = YOLO(os.path.abspath(config["cls_list"]))

flag, data = voc.parse(config["label"])

if flag == True:

    flag, data = yolo.generate(data)
    if flag == True:
        flag, data = yolo.save(data, config["output_path"], config["img_path"] ,
                               config["img_type"], config["manipast_path"])

        if flag == False:
            print("Saving Result : {}, msg : {}".format(flag, data))

    else:
        print("YOLO Generating Result : {}, msg : {}".format(flag, data))


else:
    print("VOC Parsing Result : {}, msg : {}".format(flag, data))
```

- `VOC`객체를 생성합니다.

- `YOLO`객체를 `cls_list`파일 (`*.name`)파일을 파라미터로 생성합니다.

- `voc`인스턴스를 이용하여 parsing작업을 수행합니다.

  (이 과정에서 parsing이 실패한다면 data값은 `error message`가 되며, `flag`는 `False`를 갖습니다.

  (parsing이 완료된 `data`는 `dictionary`형태의 공통 포맷입니다.)

- `yolo` 인스턴스를 이용하여 이를 yolo format으로 변경해줍니다.

- yolo format으로 된 데이터를 저장합니다.

​    

해당 내용의 사용은 3장 `Common utils`의 `dataloader`에서 사용할 예정이므로, 코드의 흐름이 맞는지 한번 직접 코드를 수정하면서 이해해보시기 바랍니다.



