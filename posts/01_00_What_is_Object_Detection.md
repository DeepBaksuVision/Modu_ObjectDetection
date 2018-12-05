Object Detection

Object Detection은 Computer Vision과 Image Processing과 관계가 깊은 컴퓨터 기술입니다. 
보통 Computer Vision에서 Detection, Recognization, Tracking 세 가지 용어가 혼재되어 사용됩니다.
Detection은 주어진 대상에서 찾고자 하는 Object의 유무를 판단하며 Recognization보다 더 작은 범위로 사용됩니다.
Recognization식은 Object가 어떤 것인지까지 구분 할 수 있고, Detection은 대상 유무만 판단합니다.
Object Recognization을 하기 위해서는 Object Detection이 선행되어야 합니다.

Object Detection 알고리즘은 찾고자하는 Object의 Feature를 사전에 추출하고 주어진 영상 내에서 Detection하는 방법이 있습니다.
전통적으로 사용하는 Feature Extraction 하는 방법은 Haar-like feature, HOG(Histogram of Oriented Gradient), SIFT(Scale Invariant Feature Transform), LBP(Local Binary Pattern), MCT(Modified Census Transform) 등이 있습니다.
Feature Extraction 후 주어진 영상 내에서 SVM(Support Vector Machine), Adaboost 등의 검출기를 이용하여 찾고자하는 객체의 특징인지 판단합니다.
최근에는 딥 러닝 중 CNN(Convolutional Neural Network)을 기반으로한 다양한 Detection 및 Reconization 알고리즘이 발전되었습니다.
본 글은 CNN을 기반으로한 YOLO(You Only Look Once)를 구현한 내용을 담고 있습니다.
