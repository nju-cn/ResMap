from collections import OrderedDict
import typing
from typing import List, Dict, Tuple

from torch import Tensor
from torch.nn import Module
from torchvision import models
from torchvision.models.detection.image_list import ImageList

from core.dnn_config import DNNConfig, BlockRule, RawLayer, InputModule, MergeModule, IMData, CpsIM, UCpsIM
from core.raw_dnn import RawDNN


class TListIM(CpsIM):
    def __init__(self, data: List[Tensor]):
        """List的长度等于图像数量"""
        super().__init__(data)

    def nzr(self) -> float:
        """非零占比，Non-Zero Rate"""
        assert len(self.data) == 1
        return float(self.data[0].count_nonzero() / self.data[0].nelement())

    def __sub__(self, other: 'TListIM') -> 'TListIM':
        assert len(self.data) == len(other.data)
        return TListIM([a - b for a, b in zip(self.data, other.data)])

    def __add__(self, other: 'TListIM') -> 'TListIM':
        assert len(self.data) == len(other.data)
        return TListIM([a + b for a, b in zip(self.data, other.data)])


class ImageListIM(CpsIM):
    def __init__(self, data: ImageList):
        """ImageList的长度等于图像数量"""
        super().__init__(data)

    def nzr(self) -> float:
        """非零占比，Non-Zero Rate"""
        assert len(self.data.tensors) == 1
        return float(self.data.tensors[0].count_nonzero() / self.data.tensors[0].nelement())

    def __sub__(self, other: 'ImageListIM') -> 'ImageListIM':
        assert self.data.image_sizes == other.data.image_sizes
        return ImageListIM(ImageList(self.data.tensors - other.data.tensors, self.data.image_sizes))

    def __add__(self, other: 'ImageListIM') -> 'ImageListIM':
        assert self.data.image_sizes == other.data.image_sizes
        return ImageListIM(ImageList(self.data.tensors + other.data.tensors, self.data.image_sizes))


class TransformWrap(Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, images: TListIM) -> ImageListIM:
        return ImageListIM(self.transform(images.data)[0])

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class ODictIM(CpsIM):
    def __init__(self, data: typing.OrderedDict[str, Tensor]):
        """OrderedDict的key为0, 1, 2, 3, pool。key的取值前后帧应该是一样的，value的前后帧应该形状是一样的"""
        super().__init__(data)

    def nzr(self) -> float:
        return sum(ts.count_nonzero()/ts.nelement() for ts in self.data.values()) / len(self.data)

    def __sub__(self, other: 'ODictIM') -> 'ODictIM':
        assert self.data.keys() == other.data.keys()
        return ODictIM(OrderedDict((k, self.data[k] + other.data[k]) for k in self.data))

    def __add__(self, other: 'ODictIM') -> 'ODictIM':
        assert self.data.keys() == other.data.keys()
        return ODictIM(OrderedDict((k, self.data[k] + other.data[k]) for k in self.data))


class BackBoneWrap(Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image_list: ImageListIM) -> ODictIM:
        features = self.backbone(image_list.data.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])
        return ODictIM(features)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        # 标记为叶子节点，避免对backbone进行展开
        return []


class RPNWrap(MergeModule):
    def __init__(self, rpn):
        super().__init__()
        self.rpn = rpn

    def forward(self, image_list: ImageListIM, features: ODictIM) -> TListIM:
        proposals, _ = self.rpn(image_list.data, features.data, None)
        return TListIM(proposals)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class DetectListIM(UCpsIM):
    def __init__(self, data: List[Dict[str, Tensor]]):
        """List长度等于输入图像数量，Dict的key为boxes, labels, scores。前后帧应该是一样的
        boxes对应的Tensor形状为(nobj, 4)，其他为nobj。前后帧的nobj可能会变化
        """
        super().__init__(data)


class RoIWrap(MergeModule):
    def __init__(self, roi_heads):
        super().__init__()
        self.roi_heads = roi_heads

    def forward(self, image_list: ImageListIM, features: ODictIM, proposals: TListIM) -> DetectListIM:
        detections, _ = self.roi_heads(features.data, proposals.data, image_list.data.image_sizes, None)
        return DetectListIM(detections)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class Post(MergeModule):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, org_images: TListIM, image_list: ImageListIM,
                detections: DetectListIM) -> DetectListIM:
        original_image_sizes = []
        for img in org_images.data:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        detections = self.transform.postprocess(detections.data, image_list.data.image_sizes, original_image_sizes)
        return DetectListIM(detections)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class RCNNRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.FasterRCNN)

    @staticmethod
    def build_dag(block: models.detection.FasterRCNN) -> List[RawLayer]:
        im = RawLayer(0, InputModule(), 'im', [])
        tfm = RawLayer(1, TransformWrap(block.transform), 'tfm', [], [im])  # images -> (images, targets)
        bkb = RawLayer(2, BackBoneWrap(block.backbone), 'bkb', [], [tfm])  # (images, targets) -> features
        rpn = RawLayer(3, RPNWrap(block.rpn), 'rpn', [], [tfm, bkb])  # (images, targets), features -> proposals
        # roi_heads: (images, targets), features, proposals -> detections
        roi = RawLayer(4, RoIWrap(block.roi_heads), 'roi', [], [tfm, bkb, rpn])
        # post: org_images, images_targets, detections -> detections
        post = RawLayer(5, Post(block.transform), 'post', [], [im, tfm, roi])
        im.ds_layers = [tfm, post]
        tfm.ds_layers = [bkb, rpn, roi, post]
        bkb.ds_layers = [rpn, roi]
        rpn.ds_layers = [roi]
        roi.ds_layers = [post]
        return [im, tfm, bkb, rpn, roi, post]


def prepare_fasterrcnn() -> DNNConfig:
    frcnn = models.detection.fasterrcnn_resnet50_fpn(True)
    frcnn.eval()
    return DNNConfig('frcnn', frcnn, {RCNNRule}, {})


if __name__ == '__main__':
    raw_dnn = RawDNN(prepare_fasterrcnn())
    for ly in raw_dnn.layers:
        print(ly)
