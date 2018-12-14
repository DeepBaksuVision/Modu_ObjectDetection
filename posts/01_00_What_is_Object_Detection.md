# Object Detection이란?

![object_detection](https://user-images.githubusercontent.com/13328380/49785835-250e0480-fd65-11e8-87b9-fd74459ade47.jpg)

​    



Object Detection은 컴퓨터비전(Computer vision), 영상처리(image processing)와 관계가 깊은 컴퓨터 기술입니다.



Computer Vision 대회에서 주로 다루는 Task들의 카테고리를 확인해보면 크게 다음과 같이 3가지로 구분될 수 있습니다.

- Classification
- Single Classification & Localization & Detection
- Multiple Object Detection & Localization & Classification

![computer_vision_task](https://user-images.githubusercontent.com/13328380/49785251-48d04b00-fd63-11e8-94ee-f9d9d9f30fe9.png)


<center>
	[컴퓨터 비전에서 다루는 3가지 주요 Task]    
</center>

​     

Computer Vision에서는 객체 검출(Object Detection), 객체 인식(Object Recognition), 객체 추적(Object Tracking) 세 가지 용어가 혼재되어 사용됩니다. 

​    

**Recognition**

- Object가 어떤 것인지 구분합니다 

**Object Detection**

- Recognition보다 더 작은 범위로써 Object의 존재 유무만 판단합니다. 

​    

Object Recognition을 하기 위해서는 해당 이미지 혹은 영상에 Object가 있고, 그것이 무엇이냐를 찾는 문제이기 때문에, Object Detection이 선행되어야 합니다.

​    

일반적으로 Object Detection 알고리즘은 찾고자 하는 Object의 특징(feature)을 사전에 추출하고 주어진 영상 내에서 해당 특징를 검출(detection)하는 접근을 주로 사용합니다.

전통적으로 영상처리에서 사용했던 Object Detection 알고리즘은 특징 엔지니어링(Feature Engineering)기법을 통하여 수학적으로 혹은 실용적으로 검증된 특징을 추출(Feature Extraction)하여 특징들의 분포(Distribution)에서 경계 결정(Boundary Decision)을 찾는 방법을 주로 사용했습니다. 전통적인 특징 추출(Feature Extraction) 방법은 Haar-like feature, HOG(Histogram of Oriented Gradient), SIFT(Scale Invariant Feature Transform), LBP(Local Binary Pattern), MCT(Modified Census Transform) 등이 있습니다.

특징 추출(Feature Extraction) 후, 특징들의 분포(Distribution)에서 경계 결정(Boundary Decision)하는 알고리즘은  SVM(Support Vector Machine), Adaboost 등의 같은 검출 알고리즘(Classifier)을 사용하여 어떤 특징(Feature)의 분포가 객체(Object)를 표현하는지 그렇지 않은지를 구분하는 것을 통해서 객체(Object)를 검출하였습니다.



![hog_sift](https://user-images.githubusercontent.com/13328380/49786073-de6cda00-fd65-11e8-94e2-ba9eea3cdad3.png)

<center>
	[기존 컴퓨터 비전에서 사용되오던 특징(Feature)들]    
</center>

​        

결론적으로 Object Detection Algorithms은 영상에서 전처리 등을 통해서 노이즈를 제거하거나, 이미지를 선명하게 만든 후에 해당 이미지에서 특징들을 추출하고, 이 특징들을 이용하여 Object Detection에 대해 분류(Classifier)하는 파이프라인(pipe line)을 따릅니다.

​    

Object Detection Algorithms에 대해서 대략적으로 요약하면 다음과 같이 작동합니다.

- 전처리 (Pre-processing)
- 특징 추출 (Feature Extraction)
- 분류 (Classifier)

​     

최근에는 딥 러닝 중 CNN(Convolutional Neural Network)을 기반으로 한 다양한 Detection 및 Recognition 알고리즘이 발전되어왔습니다. 최근 딥러닝 알고리즘에서의 Object Detection 알고리즘은 Object Detection과 Recognition을 통합하여 처리하고 있습니다. 본 ebook에서는 고전의 Object Detection Algorithms을 다루는 것보다, 현재 트렌드인 딥러닝을 이용한 Object Detection을 소개할 예정이며, 해당 Object Detection Algorithm의 구현체를 공개할 예정입니다.



![adaboost](https://user-images.githubusercontent.com/13328380/49786282-99957300-fd66-11e8-8b3d-cf87b81e59b2.png)

<center>
	[경계 결정(Boundary Decision) 알고리즘]    
</center>

​    

​    



![rcnn](https://user-images.githubusercontent.com/13328380/49786581-aa92b400-fd67-11e8-9b74-374ecc6f9740.png)    

<center>

[딥러닝 기반의 Object Detection 알고리즘 (R-CNN)]    

</center>


## Reference

[1. Evolution of Object Detection and Localization Algorithms](https://towardsdatascience.com/evolution-of-object-detection-and-localization-algorithms-e241021d8bad)

[2. Sparse Coding in a Nutshell](https://computervisionblog.wordpress.com/2014/05/24/sparse-coding-in-a-nutshell/)

[3. A Quick Guide to Boosting in ML](https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5)

[4. Comparing Classifiers](https://martin-thoma.com/comparing-classifiers/)

[5. Implementing a Soft-Margin Kernelized Support Vector Machine Binary Classifier with Quadratic Programming in R and Python](https://www.datasciencecentral.com/profiles/blogs/implementing-a-soft-margin-kernelized-support-vector-machine)

[6. A Closer Look at Object Detection, Recognition and Tracking](https://software.intel.com/en-us/articles/a-closer-look-at-object-detection-recognition-and-tracking)

[7. Wider Perspective on the Progress in Object Detection](https://techburst.io/wider-perspective-on-the-progress-in-object-detection-aac42dc98083)

[8. A Step-by-Step Introduction to the Basic Object Detection Algorithms(Part 1)](https://techburst.io/wider-perspective-on-the-progress-in-object-detection-aac42dc98083)

----

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="크리에이티브 커먼즈 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />이 저작물은 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스</a>에 따라 이용할 수 있습니다.

