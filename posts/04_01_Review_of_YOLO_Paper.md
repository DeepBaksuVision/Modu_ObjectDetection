
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

# 2. Unified Detection
 YOLO의 핵심은 `Unified Detection` 입니다. 그리고 YOLO가 `Unified Detection`가 될 수 있었던 것에는 convolutional neural network가 가장 중요한 역할을 합니다. YOLO는 단일  convolutional neural network 모델 하나로 (object detection 문제를 풀기 위한) 특징 추출, 바운딩 박스 계산, 클래스 분류를 모두 수행합니다. 즉 원본 이미지를 입력으로 받은 모델이 object detection에 필요한 모든 연산을 수행할 수 있다는 의미이며, 이것이 바로 YOLO에서 주장하는 `Unified Detection`의 개념입니다. 
 
 YOLO에는 기존의 모델에는 없었던 다양한 이점들이 있습니다. 우선 모델이 이미지 전체를 보고 바운딩 박스를 예측할 수 있습니다. 즉 모델이 이미지의 전역적인 특징을 잘 이용해서 추론을 할 수 있습니다. 그리고 `Unified Detection`이라는 용어 그대로 모델은 bounding box regression과 multi-class classification을 동시에 수행할 수 있습니다. 이러한 이점들로 인하여 YOLO는 높은 mAP를 유지하면서, end-to-end 학습이 가능하고, 실시간의 추론 속도가 가능한 것입니다.  

 지금부터는 YOLO 모델을 심도깊게 알아보도록 하겠습니다. YOLO 모델에서는 입력 이미지를 S x S 그리드로 나눕니다. 만약 어떤 객체의 중점이 특정 그리드 셀 안에 존재한다면, 해당 그리드 셀이 그 객체를 검출해야 합니다. 각 그리드 셀은 B개의 바운딩 박스를 예측합니다. 그리고 각 바운딩박스 마다 `confidence scores`를 예측하는데, `confidence scores`란 해당 바운딩 박스 내에 객체가 존재할 확률을 의미하며 0에서 1 사이의 값을 가집니다. `confidence scores`를 수식적으로 나타내면 다음과 같습니다.

 > Pr(Object) ∗ IOU_truth^pred (1)

만약 어떤 셀 안에 객체가 없으면 `confidence scores`는 0입니다. 또한 `confidence scores`는 모델이 예측한 바운딩 박스와 ground truth 간의 IOU(intersection over union)이 동일할수록 좋습니다. 

각 바운딩 박스는 값 5개를 예측합니다 : `x`, `y`, `w`, `h`, `confidence`. `(x, y)`는 바운딩 박스의 중심점이며, 각 그리드 셀마다 상대적인 값으로 표현합니다. `(w, h)`는 바운딩 박스의 width와 height인데 전체 이미지에 대해 상대적인 값으로 표현합니다. 그리고 `confidence`는 앞서 다룬 `confidence score`와 동일합니다. 

각 그리드셀마다 확률값 C를 예측합니다. C는 조건부 클래스 확률(conditional class probabilities)인 Pr(Class_i|Object)입니다. YOLO에서는 그리드 당 예측하는 바운딩 박스의 갯수(B)와 상관 없이 그리드 당 오직 하나의 클래스 확률만 예측합니다.  

Test time에서는 조건부 클래스 확률과 각 바운딩 박스는 confidence score를 곱해주며 이는 다음과 같이 나타낼 수 있습니다. 

>Pr(Classi|Object) ∗ Pr(Object) ∗ IOUtruth pred = Pr(Classi) ∗ IOUtruth pred

바운딩 박스마다 `sclass-specific confidence scores`를 얻을 수 있습니다. 이 스코어는 해당 바운딩 박스에서 특정 클래스 객체가 나타날 확률과, 객체에 맞게 바운딩 박스를 올바르게 예측했는지를 나타냅니다.    

논문에서는 PASCAL VOC를 이용하여 Evaluation을 진행합니다. 이를 위해 S = 7, B = 2, C = 20 으로 설정합니다. (PASCAL VOC에는 20개의 레이블이 있습니다). 따라서 모델의 output tensor는 7 x 7 x 30 입니다.  

## 2.1. Network Design
 앞서 말씀드렸듯이 YOLO는 convolutional neural network로 디자인된 모델입니다. YOLO는 앞단의 convolutional layers와 뒷단의 fully connected layers로 구성됩니다. convolutional layers는 이미지에서 특징을 추출하고, fully connected layers는 추출된 특징을 기반으로 클래스 확률과 바운딩 박스의 좌표를 추론합니다. YOLO는 24개의 convolutional layers와 2개의 fully connected layers로 구성됩니다. YOLO모델은 아래와 같습니다. 

 ![YOLO model](https://user-images.githubusercontent.com/13328380/48620356-3290d280-e9e3-11e8-8488-0d17e57da49b.PNG)

 논문에서는 YOLO보다 속도가 빠른 버전인 Fast YOLO도 소개합니다. Fast YOLO는 YOLO에서 단순히 레이어의 갯수를 줄인 모델인데 9개의 convolutional layer(기존 24)를 가진 모델입니다. 

## 2.2. Training
이번 장에서는 YOLO의 학습과정을 살펴보겠습니다.

### 2.2.1. Pretraining Network
 YOLO 모델의 cnvolutional layers는 ImageNet Dataset으로 Pretrain 합니다. 24개의 convolutional layers 중 앞단의 20개의 convolutional layers를 pretrain합니다. 20개의 convolutional layers 뒷단에 average-pooling layer와 fully connected layer를 붙혀서 ImageNet의 1000개의 class를 분류하는 네트워크를 만들고 이를 학습시킵니다. (논문 기준 1주일간 학습 후 ImageNet 2012 validation set 기준 top-5 accuracy 88%) YOLO는 앞단 20개의 pretrained convolutional layer에 4개의 convolutional layer와 2개의 fully connected layer를 추가하여 구성합니다. 이때 새로 추가된 레이어는 random initialized weights로 초기화합니다. 

### 2.2.2. Normalized Bounding Boxes
모델은 클래스 확률값과 바운딩 박스 좌표를 예측합니다. YOLO 에서는 정규화된(normalized) 바운딩 박스를 사용합니다. 바운딩 박스의 width와 height는 각각 이미지의 width와 height로 정규화시킵니다. 따라서 값은 0에서 1 사이의 값입니다. 바운딩 박스의 중심 좌표인 x와 y는 특정 그리드 셀에서의 offset으로 나타냅니다. 따라서 이 값도 0에서 1 사이의 값입니다. 

### 2.2.3. Nolinearity

YOLO의 activation function은 leaky ReLU를 사용하며, 단 마지막 레이어만 linear activation function을 사용합니다. leaky ReLU의 수식은 다음과 같습니다. 

![image](https://user-images.githubusercontent.com/15168540/49425352-a3582d00-f7e0-11e8-9582-719a4e84b715.png)


### 2.2.4. 고려해야 할 사항들
YOLO의 loss는 `Sum-squared error`를 기반으로 합니다. YOLO의 loss에는 바운딩 박스를 얼마나 잘 예측했는지에 대한 loss인 `localization loss`와 클래스를 얼마나 잘 예측했는지에 대한 loss인 `classification loss`가 있습니다. 이 두 개의 loss에 동일한 가중치를 할당하고 학습시키는 것은 좋지 않습니다.

또 다른 문제도 있습니다. YOLO에서는 S x S개의 그리드 셀을 예측합니다. 거의 모든 이미지에서 대다수의 그리드 셀에는 객체가 존재하지 않습니다. 이런 불균형은 YOLO가 모든 그리드 셀에서 `confidence = 0`이라고 예측하도록 학습되게 할 수 있습니다. YOLO에서는 이를 해결하기 위해서 객체가 존재하는 바운딩 박스의 `confidence loss` 가중치를 늘리고 반대로 객체가 존재하지 않는 바운딩 박스의 `confidence loss` 가중치를 줄입니다. 이는 실제로 두가지 파라미터로 조절할 수 있는데, `λ_coord`와 `λ_noobj` 입니다. 논문에서는 `λ_coord = 5`, `λ_noobj = .5`로 설정합니다. 

`Sum-squared error`를 사용하면 바운딩 박스가 큰 객체와 작은 객체에 동일한 가중치를 주는 경우에도 문제가 생길 수 있습니다. 작은 객체 바운딩박스는 조금만 어긋나도 결과에 큰 영향을 주지만 큰 객체의 바운딩 박스의 경우에는 그렇지 않습니다. 이를 해결하기 위해서 YOLO에서는 `width`와 `height`에 `square root`를 씌웁니다. 

### 2.2.5. Multiple bounding boxes per gridcell
YOLO는 하나의 그리드 셀 당 여러개의 바운딩 박스를 예측합니다. Train time에서는 객체 하나 당 하나의 바운딩 박스와 매칭시켜야 하므로, 여러개의 바운딩 박스 중 하나를 선택해야 합니다. 이를 위해(동일한 그리드 내)여러 바운딩 박스 중 gorund-truth와의 IOU가 가장 높은 하나의 바운딩 박스만 선택합니다. 이를 통해 동일한 그리드 내 여러 바운딩 박스가 서로 다른 객체의 사이즈, 종횡비(acpect ratios), 객체 클래스를 예측하게 하여 overall recall을 상승시킬 수 있습니다.    

### 2.2.6. Loss Function
Train time에 사용하는 loss function은 다음과 같습니다.  

![Objective function](https://user-images.githubusercontent.com/13328380/48620401-62d87100-e9e3-11e8-8975-93ec5b4eccf8.PNG)


여기에서 `1_obj^i` 는 i번째 그리드 셀 에 있는 j번째 바운딩박스에 객체가 존재하는지를 나타냅니다. 가령 `1_obj^i = 1`인 바운딩 박스는 해당 객체를 검출해 내야 합니다. 

수식의 `classification loss`의 경우에는 `1_obj^i = 1`인 바운딩 박스에만 적용이 되는 loss입니다(이는 Pr(Class_i|Object)를 반영한 결과입니다). 또한 `borunding box coordinate loss`의 경우에도 위와 마찬가지 입니다. 


### 2.2.7. 학습 및 하이퍼 파라미터 
논문에서는 PASCAL VOC 2007 / 2012을 이용해서 총 135 epochs를 학습시켰습니다. 하이퍼 파라미터 세팅은 다음과 같습니다. 

 `batch size = 64`  
`momentum of 0.9`  
`decay = 0.0005`


### 2.2.8 학습률 스케줄링(Learning Rate Scheduling)
YOLO의 학습률 스케줄링은 다음과 같습니다.   
>1. 첫 epoch은 `learning rate = 10^-3 에서 10^-2`까지 천천히 올립니다.  
>(처음부터 높은 learning rate를 주게되면 gradient가 발산할 수 있습니다.)
>2. 다음 75 epochs은 `learning rate = 10^-2` 으로 학습시킵니다. 
>3. 다음 30 epochs는 `learning rate = 10^-3` 으로 학습시킵니다. 
>4. 다음 30 epochs는 `learning rate = 10^-4` 으로 학습시킵니다. 

### 2.2.9 오버피팅 방지 
YOLO에서는 오버피팅을 방지하기 위해 dropout과 data augmentation 기법을 사용합니다. dropout은 첫 번째 fully connected layer에 붙으며 `dropout rate = 0.5`로 설정했습니다. Data augmentation에는 `scaling`, `translation`, `exposure`, `saturation`을 조절하는 방식으로 다양하게 진행하며, `scaling`과 `translation`은 원본 이미지 사이즈의 20%까지 임의로 조절하며, `scaling`, `translation`은 HSV 공간에서 1.5배까지 임의로 조절합니다. 


## 2.3. 추론(Inference)
모델은 이미지 한장 당 총 98개의 바운딩 박스를 예측합니다. 각 바운딩 박스에는 클래스 확률이 있습니다. 입력 이미지를 네트워크에 단 한번만 통과시키면 되므로 Test time시 YOLO의 속도는 상당히 빠릅니다. YOLO과 같이 그리드 셀을 이용한 디자인으로 겪을 수 있는 문제점 하나가 있습니다. 그것은 바로 하나의 객체를 여러 그리드 셀이 동시에 검출하는 경우입니다. 특히 객체가 그리드셀들의 경계에 위치하거나 여러 그리드 셀을 포함할 만큼 큰 객체인 경우 이런 현상이 자주 발생합니다. 비최대억제(Non-maximal suppression)는 그러한 다중 검출 문제를 해결할 수 있는 좋은 방법입니다. YOLO는 NMS를 통해 mAP를 2-3% 가량 올릴 수 있었습니다.

## 2.4. YOLO의 한계 
YOLO 모델의 한계점들이 몇 가지 있습니다. 우선 YOLO는 각 그리드 셀마다 오직 하나의 객체만을 검출할 수 있습니다. 이는 객체 검출에서 아주 강한 공간적 제약(spatial constraints)입니다. 이러한 공간적 제약으로 인해 YOLO는 '새 떼'와 같이 작은 객체들이 무리지어 있는 경우의 객체 검출이 제한적일 수 있습니다. 그리고 바운딩 박스를 데이터로부터 학습하기 때문에 일반화 능력이 떨어지고, 이로 인해 train time에 보지 못했던 종횡비의 객체를 잘 검출하지 못합니다. 그리고 마지막으로 YOLO에서 가장 문제가 되는 부분이 바로 잘못된 localizations 입니다. 


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

