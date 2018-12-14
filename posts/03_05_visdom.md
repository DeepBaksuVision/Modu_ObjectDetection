# 학습 과정 시각화하기
이번 장에서는 PyTorch에서 학습 과정을 시각화할 수 있는 툴에 대해서 알아보겠습니다. Tensorflow의 Tensorboard처럼 PyTorch에서도 사용할 수 있는 유용한 도구들이 많습니다. 이번에는 Visdom에서 알아보겠습니다.  

## 01. Visdom
Visdom은 PyTorch에서 사용할 수 있는 시각화 도구 중 하나입니다.

![visdom](https://camo.githubusercontent.com/d69475a01f9f327fc42931a21df8134d1fbdfc19/68747470733a2f2f6c68332e676f6f676c6575736572636f6e74656e742e636f6d2f2d62714839555843772d42452f574c3255736472726241492f41414141414141416e59632f656d727877436d6e7257345f434c54797955747442305359524a2d693443436951434c63422f73302f53637265656e2b53686f742b323031372d30332d30362b61742b31302e35312e30322b414d2e706e67253232766973646f6d5f626967253232)

Visdom의 설치는 [Visdom 문서](https://github.com/facebookresearch/visdom#setup) 를 확인바랍니다. Visdom은 자세한 API를 제공하지 않습니다. 하지만 API가 별도로 존재하지 않는 이유는 사용법이 간단하기 때문이기도 합니다.  Visdom의 Github repository에서 제공하는 [간단한 API문서](https://github.com/facebookresearch/visdom#api)와 [예제코드](https://github.com/facebookresearch/visdom/blob/master/example/demo.py)를 숙지하면 누구나 바로 자신의 프로젝트에 Visdom을 추가할 수 있습니다.

## 02. 프로젝트에 적용
이제 본격적으로 우리 프로젝트에 적용해봅시다. YOLO와 같은 multi-task learning은 계산해야 할 로스가 많습니다. 따라서 전체 로스만 시각화하는 것도 좋지만 각 로스들을 개별적으로 출력해 줄 필요가 있습니다.

```python
if USE_VISDOM:
    viz = visdom.Visdom(use_incoming_socket=False)
    vis_title = 'Yolo V1 Deepbaksu_vision (feat. martin, visionNoob) PyTorch on ' + 'VOC'
    vis_legend = ['Train Loss']
    iter_plot = create_vis_plot(viz, 'Iteration', 'Total Loss', vis_title, vis_legend)
    coord1_plot = create_vis_plot(viz, 'Iteration', 'coord1', vis_title, vis_legend)
    size1_plot = create_vis_plot(viz, 'Iteration', 'size1', vis_title, vis_legend)
    noobjectness1_plot = create_vis_plot(viz, 'Iteration', 'noobjectness1', vis_title, vis_legend)
    objectness1_plot = create_vis_plot(viz, 'Iteration', 'objectness1', vis_title, vis_legend)
    obj_cls_plot = create_vis_plot(viz, 'Iteration', 'obj_cls', vis_title, vis_legend)
};
```  
여기에서는 visdom 객체를 생성합니다. 그 다음 사용법은 matplotlib와 같은 다른 시각화 툴과 아주 유사합니다. title, legend를 정의하고, 각 로스를 출력한 plot 객체를 선언합니다. plot 객체를 create_vis_plot() 함수를 사용합니다.   

![image](https://user-images.githubusercontent.com/15168540/49014603-08e66100-f1c4-11e8-9fdc-bbf79db994b0.png)
create_visdom() 함수를 실행하면 다음과 같이 visdom 서버에 plot이 추가됩니다.

```python
if USE_VISDOM:
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), loss.item(), iter_plot, None, 'append')
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_coord1_loss, coord1_plot, None, 'append')
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_size1_loss, size1_plot, None, 'append')
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), obj_class_loss, obj_cls_plot, None, 'append')
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), noobjness1_loss, noobjectness1_plot, None, 'append')
    update_vis_plot(viz, (epoch + 1) * total_step + (i + 1), objness1_loss, objectness1_plot, None, 'append')
```
그리고 학습 과정을 Visdom에 출력하려면 다음과 같이 계산한 loss를 앞서 선언한 plot 객체에 뿌려주기만 하면 됩니다. 여기에는 update_vis_plot() 함수를 사용합니다.

![image](https://user-images.githubusercontent.com/15168540/49014571-ebb19280-f1c3-11e8-965f-0ef3100bb977.png)
update_vis_plot()을 실행하면 Visdom 서버에서 loss가 지속적으로 업데이트됩니다.


# REFERENCES
[1] https://github.com/facebookresearch/visdom
