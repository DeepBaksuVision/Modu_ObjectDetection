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