from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
#类的实例化，Registry是一个类，传入的是一个字符串。该字符串为Registry类的name属性值