# 1. Object Detection이란?

![object_detection](https://user-images.githubusercontent.com/13328380/49785835-250e0480-fd65-11e8-87b9-fd74459ade47.jpg)

​    



Object Detection은 Computer Vision과 Image Processing과 관계가 깊은 컴퓨터 기술입니다. 



Computer Vision 대회에서 주로 다루는 Task들의 카테고리를 확인해보면 크게 다음과 같이 3가지로 구분될 수 있습니다.

- Classification
- Single Classification & Localization & Detection
- Multiple Object Detection & Localization & Classification

![computer_vision_task](https://user-images.githubusercontent.com/13328380/49785251-48d04b00-fd63-11e8-94ee-f9d9d9f30fe9.png)

​    

Computer Vision에서는 Detection, Recognization, Tracking 세 가지 용어가 혼재되어 사용됩니다. 여기서 Object Detection이란 주어진 대상에서 찾고자 하는 Object의 유무만 판단하며, Recognization보다 더 작은 범위로 사용됩니다. Recognization식은 Object가 어떤 것인지까지 구분 할 수 있고, Object Detection은 대상 유무만 판단합니다. Object Recognization을 하기 위해서는 Object Detection이 선행되어야 합니다. 



Object Detection 알고리즘은 찾고자하는 Object의 Feature를 사전에 추출하고 주어진 영상 내에서 해당 Feature를 Detection하는 접근을 주로 사용하게 됩니다. 

전통적으로 사용하는 Feature Extraction 하는 방법은 Haar-like feature, HOG(Histogram of Oriented Gradient), SIFT(Scale Invariant Feature Transform), LBP(Local Binary Pattern), MCT(Modified Census Transform) 등이 있습니다.



![hog_sift](https://user-images.githubusercontent.com/13328380/49786073-de6cda00-fd65-11e8-94e2-ba9eea3cdad3.png)

​    

Feature Extraction 후, 주어진 영상 내에서 얻은 Feature를 이용하여 SVM(Support Vector Machine), Adaboost 등의 같은 검출기(Classifier)를 이용하여 찾고자하는 객체의 특징인지 판단합니다.



결론적으로 Object Detection Algorithms은 영상에서 노이즈나, 이미지를 선명하게 만든 후에 해당 이미지에서 특징들을 추출하고, 이 특징들을 이용하여 Object Detection에 대해 분류(Classifier)하는 것으로 작동합니다.



Object Detection Algorithms에 대해서 대략적으로 요약하면 다음과 같이 작동합니다.

- pre-processing
- Feature Extraction
- Classifier



최근에는 딥 러닝 중 CNN(Convolutional Neural Network)을 기반으로한 다양한 Detection 및 Reconization 알고리즘이 발전되어왔습니다. 본 ebook에서는 고전의 Object Detection Algorithms을 다루는 것보다, 현재 트렌드인 딥러닝을 이용한 Object Detection에 대해서 소개할 예정이며, 해당 Object Detection Algorithm의 구현체를 공개할 예정입니다. 



![adaboost](https://user-images.githubusercontent.com/13328380/49786282-99957300-fd66-11e8-8b3d-cf87b81e59b2.png)







![rcnn](https://user-images.githubusercontent.com/13328380/49786581-aa92b400-fd67-11e8-9b74-374ecc6f9740.png)

​    

## Reference

[1. Evolution of Object Detection and Localization Algorithms](https://towardsdatascience.com/evolution-of-object-detection-and-localization-algorithms-e241021d8bad)

[2. Sparse Coding in a Nutshell](https://computervisionblog.wordpress.com/2014/05/24/sparse-coding-in-a-nutshell/)

[3. A Quick Guide to Boosting in ML](https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5)

[4. Comparing Classifiers](https://martin-thoma.com/comparing-classifiers/)

[5. Implementing a Soft-Margin Kernelized Support Vector Machine Binary Classifier with Quadratic Programming in R and Python](https://www.datasciencecentral.com/profiles/blogs/implementing-a-soft-margin-kernelized-support-vector-machine)

[6. A Closer Look at Object Detection, Recognition and Tracking](https://software.intel.com/en-us/articles/a-closer-look-at-object-detection-recognition-and-tracking)

[7. Wider Perspective on the Progress in Object Detection](https://techburst.io/wider-perspective-on-the-progress-in-object-detection-aac42dc98083)

[8. A Step-by-Step Introduction to the Basic Object Detection Algorithms(Part 1)](https://techburst.io/wider-perspective-on-the-progress-in-object-detection-aac42dc98083)
