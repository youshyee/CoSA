from .resnet38_seg import Net as WRN38
from .resnet38d import Net as WRN38_cls
from .deeplabv3plus import DeepLabV3Plus
__all__ = ['WRN38', 'WRN38_cls', 'DeepLabV3Plus']
