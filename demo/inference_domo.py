
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
config_file = '/remote-home/gyf/yzm/mmdet/mmdetection-master/configs/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/remote-home/gyf/yzm/mmdet/mmdetection-master/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:7')

img = '/remote-home/gyf/yzm/mmdetection/demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(img, result, model.CLASSES)
print('done')