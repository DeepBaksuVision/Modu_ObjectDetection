# 01). You Only Look Once - Paper Review



## 1. Introduction

해당 논문이 나오기 전(2015년 전)에는 딥러닝 기반의 Object Detection System들은 Classification 모델을 Object Detection 시스템에 맞게 변형한 모델들이 주를 이루었습니다. Classification 모델을 변형한 시스템은 다양한 위치 및 크기에 대해서 학습하고 테스트를 진행했습니다. 

대부분의 변형된 모델들은 [Deformable Parts Models(DPM)](https://www.cs.cmu.edu/~deva/papers/dpm_acm.pdf)이 sliding window 방법론을 사용하여 전체 이미지를 Classifier로 스캔하는 방식 형태의 방법론을 사용했습니다.

​    

<p align="center"><img src="https://user-images.githubusercontent.com/13328380/49448628-a96bff00-f81c-11e8-8d54-3435f1c2a3e4.gif" /></p>

<center>
    <a href="https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/">
	[sliding window approach]    
    </a>
</center>

​    

제일 최근(2015년)의 모델로는 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)이 있는데, [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)은 bounding box일 확률이 높은 곳을 제안하고 제안된 box영역을 Classifier로 넣어 classification을 수행합니다. 그 다음 후처리를 통해서 bounding box를 개선(합치고, 제거하고)하는 방식을 사용합니다. 

[R-CNN](https://arxiv.org/pdf/1311.2524.pdf)모델은 region proposal  / classification / box regression라는 3가지  단계를 거치는 과정을 가지며, 3가지 단계를 개별적으로 학습해야하므로 복잡한 파이프라인을 갖게됩니다. [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)은 이런 복잡한 파이프라인 때문에 최적화에 어렵고 굉장히 큰 inference time을 갖는다는 단점이 있습니다.

​    

![R-CNN](https://user-images.githubusercontent.com/13328380/49448834-0962a580-f81d-11e8-9926-e6cd2a5fe646.jpg)

<center>
    <a href="https://jamiekang.github.io/2017/05/28/faster-r-cnn/">
	[R-CNN]    
    </a>
</center>

​    

You Only Look Once(이하 YOLO)는 기존의 방법론인 Classification모델을 변형한 방법론에서 벗어나, Object Detection 문제를 regression문제로 정의하는 것을 통해서 이미지에서 직접적으로 bounding box 좌표와 각 클래스의 확률을 구합니다. 

YOLO는 end-to-end 방식의 통합된 구조를 가지고 있으며, 이미지를 convolutional neural network에 한번 평가(inference)하는 것을 통해서 동시에 다수의 bounding box와 class 확률을 구하게됩니다. 이렇게 통합된 모델로 인해서  YOLO는 몇가지 장점을 가지게 됩니다.

첫번째로 Titan X에서 45fps를 달성하며 빠른버전의 경우 150fps의 빠른 성능을 자랑함과 동시에 다른 실시간 시스템 대비 2배 이상의 mAP(mean average precision)를 높은 성능을 보여줍니다.

두번째로는. sliding window방식이 아닌 convolutional neural network를 사용하는 것으로 인해 전체 이미지를 보게끔 유도되어(문맥정보;contextual information)각 class에 대한 표현을 더 잘 학습하게됩니다. ([R-CNN](https://arxiv.org/pdf/1311.2524.pdf)의 경우에는 이미지의 백그라운드에 의해서 Object Detection을 실패하는 경우가 있습니다.)

세번째로는 일반화된 Object의 표현을 학습합니다. 실험적으로 자연의 dataset을 학습시킨 이후에 학습시킨 네트워크에 artwork 이미지를 입력했을 때, [DPM](https://www.cs.cmu.edu/~deva/papers/dpm_acm.pdf), [R-CNN](https://arxiv.org/pdf/1311.2524.pdf) 대비 많은 격차로 좋은 Detection 성능을 보여줍니다.

​    

YOLO에 대한 특징을 정리해보자면 다음과 같습니다.

- state-of-the-art에는 못미치는 성능
- 빠른 inference
- 작은 물체에 대해서 Detection 성능이 떨어짐
- end-to-end 형태의 inference와 train
- 좋은 일반화 성능



## 3. Experiments



### 1). Comparison to Other Real-Time Systems

DPM 100Hz/30Hz 구현체와 GPU 버전의 YOLO를 기준으로 모델을 YOLO 내의 변형 모델의 성능 비교를 합니다.



#### Fast YOLO

- PASCAL VOC기준 가장 빠른 Object Detection 알고리즘
- 52.7% mAP (100Hz/30Hz DPM 기준 2배 이상 향상)



#### YOLO

- 실시간 성능을 만족하면서 mAP는 63.4%
- VGG-16을 feature extractor로 사용한 경우에는 성능이 증가하나, 속도 감소



#### [Fastest DPM](http://www.cbsr.ia.ac.cn/users/jjyan/Fastest_DPM.pdf)

- 약간의 mAP 성능하락으로 Detection 속도를 높인 모델

  하지만 실시간 detection system이라고 부르기엔 속도가 느림

- 30.7% mAP로 성능 및 속도측면에서 YOLO 대비 전반적으로 안좋음



### R-CNN 계열

최근 [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)계열에서 속도가 개선된 [Fast R-CNN](https://arxiv.org/pdf/1506.01497.pdf)이 나왔으나, 속도 측면에서 실시간 성능에는 한참 못미치는 성능을 갖음

- [Fast R-CNN](https://arxiv.org/pdf/1506.01497.pdf) : 70% mAP, 0.5 fps
- [Fast R-CNN](https://arxiv.org/pdf/1506.01497.pdf) with VGG-16 : 73.2% mAP, 7 fps

​    

![benchmark_with_other_systems](https://user-images.githubusercontent.com/13328380/49447560-33669880-f81a-11e8-9a2f-db7588e8357b.png)

​    

### 2) VOC 2007 Error Analysis

VOC 2007 Dataset에서 YOLO, [Fast R-CNN](https://arxiv.org/pdf/1506.01497.pdf) 모델의 에러분석을 하기 위하여 다음과 같은 레퍼런스 [Diagnosing Error in Object Detectors](http://dhoiem.web.engr.illinois.edu/publications/eccv2012_detanalysis_derek.pdf) 를 참고하여 기준을 잡음



- Correct : class가 정확하며 IOU가 0.5보다 큰 경우
- Localization : class가 정확하고, IOU가 0.1보다 크고 0.5보다 작은 경우
- Similar : class가 유사하고 IOU가 0.1보다 큰 경우
- Other : class는 틀렸으나, IOU가 0.1보다 큰 경우
- Background : 어떤 Object이던간에 IOU가 0.1보다 작은 경우



분석 결과는 다음과 같음

- YOLO는 localize 에러가 심함
- [Fast R-CNN](https://arxiv.org/pdf/1506.01497.pdf)은 상대적으로 localization 에러는 적으나 background 에러가 큼.

​    

![error_analysis](https://user-images.githubusercontent.com/13328380/49448061-64939880-f81b-11e8-85d8-7bfea72dae47.png)

​    

### 3). VOC 2012 Result



VOC 2012 결과에서 YOLO는 57.9% mAP를 달성함

​    

![pascal_voc2012_leaderboard](https://user-images.githubusercontent.com/13328380/49448183-af151500-f81b-11e8-8c2f-a3eefaf96f24.png)

​    

### 4). Generalizability: Person Detection in Artwork

[Picasso Dataset](https://people.eecs.berkeley.edu/~shiry/publications/Picasso_ECCV_2014.pdf) 와 [People-Art Dataset](https://core.ac.uk/download/pdf/38151134.pdf)를 이용하여 YOLO와 다른 Detection System들의 성능을 비교함

​    

![generalization_result](https://user-images.githubusercontent.com/13328380/49448429-3bbfd300-f81c-11e8-9e87-ad2ef8923977.png)

​    

## 4. Qualitative Result

아래 사진은 YOLO의 Object Detection 결과입니다.

​    

![qualitative_result](https://user-images.githubusercontent.com/13328380/49448530-70338f00-f81c-11e8-9bcc-0842659bc1c0.png)

​    

## Reference

아래 자료들은 본 문서와 함께 보시면 YOLO를 이해하는데 도움이 될 수 있는 자료 모음입니다.



#### 1). CVPR - You Only Look Once: Unified, Real-Time Object Detection

[![You Only Look Once: Unified, Real-Time Object Detection)](https://img.youtube.com/vi/NM6lrxy0bxs/0.jpg)](https://www.youtube.com/watch?v=NM6lrxy0bxs) 



#### 2). You Only Look Once

[![YOLO: You only look once (How it works)](https://img.youtube.com/vi/L0tzmv--CGY/0.jpg)](https://www.youtube.com/watch?v=L0tzmv--CGY) 



#### 3). PR-016: You only look once

[![PR-016: You only look once: Unified, real-time object detection](https://img.youtube.com/vi/eTDcoeqj1_w/0.jpg)](https://www.youtube.com/watch?v=eTDcoeqj1_w) 



- [Curt-Park - [분석] YOLO](https://curt-park.github.io/2017-03-26/yolo/)
- [Jamie Kang's - You Only Look Once : Unified, Real-Time Object Detection](https://jamiekang.github.io/2017/06/18/you-only-look-once-unified-real-time-object-detection/)
- [모두의 연구소 - YOLO 논문요약](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=5&ved=2ahUKEwiozv-VtobfAhXDdN4KHRFND4AQFjAEegQIAxAC&url=http%3A%2F%2Fwww.modulabs.co.kr%2F%3Fmodule%3Dfile%26act%3DprocFileDownload%26file_srl%3D20615%26sid%3D8629ec2e16ef451a8ce8ad206b112b42%26module_srl%3D18164&usg=AOvVaw0IGwOXa1Er1GIlNH7o0DVi)
- [sogangori - YOLO, Object Detection Network](http://blog.naver.com/PostView.nhn?blogId=sogangori&logNo=220993971883&parentCategoryNo=15&categoryNo=&viewDate=&isShowPopularPosts=true&from=search)
- [Arc Lab - [논문 요약 12] You Only Look Once: Unified, Real-Time Object Detection](http://arclab.tistory.com/167)
- [Wonju Seo - You Only Look Once : Unified Real-Time Object Detection](http://wewinserv.tistory.com/79)

