from collections import OrderedDict
import typing
from typing import List, Dict, Tuple

from torch import Tensor
from torch.nn import Module
from torchvision import models
from torchvision.models.detection.image_list import ImageList

from core.dnn_config import DNNConfig, BlockRule, RawLayer, InputModule, MergeModule
from core.raw_dnn import RawDNN


class BackBoneWrap(Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images_targets: Tuple[ImageList, None]) -> typing.OrderedDict[str, Tensor]:
        features = self.backbone(images_targets[0].tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])
        return features

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        # 标记为叶子节点，避免对backbone进行展开
        return []


class RPNWrap(MergeModule):
    def __init__(self, rpn):
        super().__init__()
        self.rpn = rpn

    def forward(self, images_targets: Tuple[ImageList, None], features: typing.OrderedDict[str, Tensor]) -> List[Tensor]:
        proposals, _ = self.rpn(images_targets[0], features, images_targets[1])
        return proposals

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class RoIWrap(MergeModule):
    def __init__(self, roi_heads):
        super().__init__()
        self.roi_heads = roi_heads

    def forward(self, images_targets: Tuple[ImageList, None], features: typing.OrderedDict[str, Tensor],
                proposals: List[Tensor]) -> List[Dict[str, Tensor]]:
        images, targets = images_targets
        detections, _ = self.roi_heads(features, proposals, images.image_sizes, targets)
        return detections

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class Post(MergeModule):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, org_images: List[Tensor], images_targets: Tuple[ImageList, None],
                detections: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        original_image_sizes = []
        for img in org_images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        return self.transform.postprocess(detections, images_targets[0].image_sizes, original_image_sizes)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class RCNNRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.FasterRCNN)

    @staticmethod
    def build_dag(block: models.detection.FasterRCNN) -> List[RawLayer]:
        im = RawLayer(0, InputModule(), 'im', [])
        tfm = RawLayer(1, block.transform, 'tfm', [], [im])  # images -> (images, targets)
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
