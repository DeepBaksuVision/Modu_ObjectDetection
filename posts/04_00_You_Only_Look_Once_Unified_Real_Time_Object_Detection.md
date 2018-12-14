# YOLO: Real-Time Object Detection
YOLO(You Only Look Once: Unified, Real-Time Object Detection)는 대표적인 Single Shot 계열의 Object Detection 모델 중 하나입니다. 제법 높은 성능과, (실시간 처리가 가능한) 모델의 빠른 연산속도로 지금도 아주 많은 사랑을 받고 있습니다. CVPR 2016에 발표된 [YOLOv1](https://arxiv.org/abs/1506.02640)을 시작으로(CVPR2016에서 저자가 보여준 [실시간 데모](https://youtu.be/NM6lrxy0bxs?t=675)가 압권입니다), 현재는 [YOLOv2](https://arxiv.org/abs/1612.08242), 그리고 [YOLOv3](https://arxiv.org/abs/1804.02767)까지 버전이 업데이트되었습니다. 이번 포스트에서는 YOLO의 가장 기본 모델인 YOLOv1에 대해서 알아보도록 하겠습니다.



**실제 코드는 [DeepBaksuVision/You_Only_Look_Once](https://github.com/DeepBaksuVision/You_Only_Look_Once)에서 확인할 수 있습니다.**




**동영상 1 . YOLO Object Detection(YOLOv1)**

[![You Only Look Once: Unified, Real-Time Object Detection)](https://img.youtube.com/vi/EJy0EI3hfSg/0.jpg)](https://www.youtube.com/watch?v=EJy0EI3hfSg)

---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="크리에이티브 커먼즈 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />이 저작물은 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">크리에이티브 커먼즈 저작자표시-비영리-동일조건변경허락 4.0 국제 라이선스</a>에 따라 이용할 수 있습니다.

