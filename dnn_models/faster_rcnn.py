from collections import OrderedDict
import typing
from typing import List, Dict, Tuple, Any

from torch import Tensor, jit
from torch.nn import Module, functional
import torch.nn as nn
from torchvision import models, ops
from torchvision.models.detection.image_list import ImageList
from torchvision.models.resnet import Bottleneck
from torchvision.models.detection.rpn import permute_and_flatten, concat_box_prediction_layers, RegionProposalNetwork

from core.dnn_config import DNNConfig, BlockRule, RawLayer, InputModule, MergeModule, ForkModule, IMData, TensorIM, \
    CpsIM, UCpsIM, BasicFork
from core.raw_dnn import RawDNN

from copy import deepcopy

'''
中间数据类型
'''
class TListIM(CpsIM):
    def __init__(self, data: List[Tensor]):  # or List[TensorIM]
        """List的长度等于图像数量"""
        super().__init__(data)

    def nzr(self) -> float:
        """非零占比，Non-Zero Rate"""
        return sum(ts.count_nonzero() for ts in self.data) / sum(ts.nelement() for ts in self.data)

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


class ODictIM(CpsIM):
    def __init__(self, data: typing.OrderedDict[str, Tensor]):  # or OrderedDict[str, TensorIM]
        """OrderedDict的key为0, 1, 2, 3, pool。key的取值前后帧应该是一样的，value的前后帧应该形状是一样的"""
        super().__init__(data)

    def nzr(self) -> float:
        return sum(ts.count_nonzero() for ts in self.data.values()) / sum(ts.nelement() for ts in self.data.values())

    def __sub__(self, other: 'ODictIM') -> 'ODictIM':
        assert self.data.keys() == other.data.keys()
        return ODictIM(OrderedDict((k, self.data[k] - other.data[k]) for k in self.data))

    def __add__(self, other: 'ODictIM') -> 'ODictIM':
        assert self.data.keys() == other.data.keys()
        return ODictIM(OrderedDict((k, self.data[k] + other.data[k]) for k in self.data))


class DetectListIM(UCpsIM):
    def __init__(self, data: List[Dict[str, Tensor]]):
        """List长度等于输入图像数量，Dict的key为boxes, labels, scores。前后帧应该是一样的
        boxes对应的Tensor形状为(nobj, 4)，其他为nobj。前后帧的nobj可能会变化
        """
        super().__init__(data)


class StringListIM(UCpsIM):
    def __init__(self, data: List[str]):
        '''用于图像金字塔中的names数据'''
        super().__init__(data)


class TupleTListIM(CpsIM):
    def __init__(self, data: Tuple[List[Tensor]]):
        '''由多个List[Tensor]组成的Tuple（也可以是List）的中间数据格式'''
        super().__init__(data)

    def nzr(self) -> float:
        nonzero = 0
        nelement = 0
        for lst in self.data:
            nonzero += sum(ts.count_nonzero() for ts in lst)
            nelement += sum(ts.nelement() for ts in lst)
        return nonzero / nelement

    def __sub__(self, other: 'TupleTListIM') -> 'TupleTListIM':
        assert len(self.data) == len(other.data)
        res = []
        for lst1, lst2 in zip(self.data, other.data):
            res.append([a - b for a, b in zip(lst1, lst2)])
        res = tuple(res)
        return TupleTListIM(res)

    def __add__(self, other: 'TupleTListIM') -> 'TupleTListIM':
        assert len(self.data) == len(other.data)
        res = []
        for lst1, lst2 in zip(self.data, other.data):
            res.append([a + b for a, b in zip(lst1, lst2)])
        res = tuple(res)
        return TupleTListIM(res)


class TupleIM(UCpsIM):
    '''一个函数可能有多个返回值，需要将它们组合成TupleIM，再通过单独的Module提取各部分'''
    def __init__(self, data: Tuple[Any]):
        super().__init__(data)


class ImageShapeIM(UCpsIM):
    '''RoI中存储image_shape，即List[Tuple[int, int]]，指示输入图片的大小，不必压缩'''
    def __init__(self, data: List[Tuple[int]]):
        super().__init__(data)


'''
用来对模型进行包裹，对数据进行类型转换和预处理的Module Wrap
'''


class TransformWrap(ForkModule):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, images: TListIM) -> ImageListIM:
        return ImageListIM(self.transform(images.data)[0])

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return []


class BackBoneWrap(ForkModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image_list: ImageListIM) -> ODictIM:
        features = self.backbone(image_list.data.tensors)
        if isinstance(features, Tensor):
            features = OrderedDict([('0', features)])
        return ODictIM(features)

    # def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
    #     # 标记为叶子节点，避免对backbone进行展开
    #     return []


class RPNWrap(MergeModule):
    def __init__(self, rpn):
        super().__init__()
        self.rpn = rpn

    def forward(self, image_list: ImageListIM, features: ODictIM) -> TListIM:
        proposals, _ = self.rpn(image_list.data, features.data, None)
        return TListIM(proposals)

    # def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
    #     return []


class RoIWrap(MergeModule):
    def __init__(self, roi_heads):
        super().__init__()
        self.roi_heads = roi_heads

    def forward(self, image_list: ImageListIM, features: ODictIM, proposals: TListIM) -> DetectListIM:
        detections, _ = self.roi_heads(features.data, proposals.data, image_list.data.image_sizes, None)
        return DetectListIM(detections)

    # def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
    #     return []


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


class BodyWrap(Module):
    def __init__(self, body: models._utils.IntermediateLayerGetter):
        super().__init__()
        self.body = body

    def forward(self, image_tensor: TensorIM) -> ODictIM:
        features = self.body(image_tensor.data)
        for k, v in features.items():
            features[k] = TensorIM(v)
        return ODictIM(features)

    # def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
    #     return []


class FPNWrap(Module):
    def __init__(self, fpn: ops.FeaturePyramidNetwork):
        super().__init__()
        self.fpn = fpn

    def forward(self, features: ODictIM) -> ODictIM:  # OrderedDict([str, TensorIM]) -> OrderedDict([str, Tensor])
        odict = deepcopy(features.data)
        for k, v in odict.items():
            odict[k] = v.data
        fpn_features = self.fpn(odict)
        if isinstance(fpn_features, Tensor):
            fpn_features = OrderedDict([('0', features)])
        return ODictIM(fpn_features)  # OrderedDict([str, Tensor])

    # def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
    #     return []


class ConvertWrap(Module):
    '''Tensor->Tensor的Module的通用类型转换Wrap，将TensorIM转换为Tensor后交给模型推断，再将结果转换成TensorIM'''
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x: TensorIM) -> TensorIM:
        x = self.module(x.data)
        return TensorIM(x)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return self.module.named_children()


class InterpolateWrap(MergeModule):
    def forward(self, last_inner: TensorIM, inner_lateral: TensorIM):
        feat_shape = inner_lateral.data.shape[-2:]
        inner_top_down = functional.interpolate(last_inner.data, size=feat_shape, mode="nearest")
        return TensorIM(inner_top_down)


class ExtraBlockWrap(Module):
    def __init__(self, extra_block: ops.feature_pyramid_network.ExtraFPNBlock):
        super().__init__()
        self.extra_block = extra_block

    def forward(self, names: StringListIM, x: TListIM, results: TListIM) -> ODictIM:  # OrderedDict[str, Tensor]
        # 将IM数据拆成真正数据
        results_lst = results.data
        x_lst = x.data
        # 推断
        new_results, new_names = self.extra_block(results_lst, x_lst, names.data)
        # 将数据组织为ODict(OrderedDict[str, Tensor])后返回
        out = OrderedDict([(k, v) for k, v in zip(new_names, new_results)])
        return ODictIM(out)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        '''extra_block有可能可以继续展开，但层数很少，故设置为不再展开'''
        return []


class RPNAnchorGenWrap(MergeModule):
    '''通过images, features生成anchor: TListIM'''
    def __init__(self, anchor_gen: models.detection.anchor_utils.AnchorGenerator):
        super().__init__()
        self.anchor_gen = anchor_gen

    def forward(self, images: ImageListIM, features: TListIM):
        anchors = self.anchor_gen(images.data, features.data)
        return TListIM(anchors)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return self.anchor_gen.named_children()


class RPNCalculateWrap(MergeModule):
    '''RPN中RPNHead之后的额外计算'''
    def __init__(self, box_coder, filter_proposals):
        '''将rpn自带的一些模块引入'''
        super().__init__()
        self.box_coder = box_coder
        self.filter_proposals = filter_proposals

    def forward(self, images: ImageListIM, obj_pred: TupleTListIM, anchors: TListIM) -> TListIM:
        images = images.data
        objectness, pred_bbox_deltas = obj_pred.data
        anchors = anchors.data
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        return TListIM(boxes)


class BoxRoIPoolWrap(Module):
    '''RoI中的box_roi_pool部分'''
    def __init__(self, box_roi_pool: ops.poolers.MultiScaleRoIAlign):
        super().__init__()
        self.box_roi_pool = box_roi_pool

    def forward(self, img_feature_prop: TupleIM) -> TensorIM:
        image_shapes = img_feature_prop.data[0].data.image_sizes
        features = img_feature_prop.data[1].data
        proposals = img_feature_prop.data[2].data
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        return TensorIM(box_features)

    def named_children(self) -> typing.Iterator[Tuple[str, 'Module']]:
        return self.box_roi_pool.named_children()


class RoICalculateWrap(MergeModule):
    '''RoI中一系列对predictor输出的后续处理'''
    def __init__(self, postprocess_detections: typing.Callable):
        super().__init__()
        self.postprocess_detections = postprocess_detections

    def forward(self, image_shapes: ImageShapeIM, proposals: TListIM, class_and_box: TListIM):
        image_shapes = image_shapes.data
        proposals = proposals.data
        class_logits, box_regression = class_and_box.data

        result = jit.annotate(List[Dict[str, Tensor]], [])
        boxes, scores, labels = \
            self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        return DetectListIM(result)


'''
非模型本身含有的，为方便数据处理而自行添加的Module
'''


class ImgList2Tensor(Module):
    '''body中需要先将输入的ImageListIM转换为TensorIM，再交给后续模型'''
    def forward(self, x: ImageListIM) -> TensorIM:
        x = x.data.tensors
        return TensorIM(x)


class MergeFeatures(MergeModule):
    '''body中最后将多个features整合为OrderedDict'''
    def __init__(self, out_names):
        super().__init__()
        self.out_names = out_names

    def forward(self, *features: TensorIM) -> ODictIM:  # [str, TensorIM]
        assert len(features) == len(self.out_names)
        out = OrderedDict([(k, v.data) for (k, v) in zip(self.out_names, features)])
        return ODictIM(out)


class BottleneckAdd(MergeModule):
    """Bottleneck末尾合并各分支的模块"""

    def forward(self, *inputs: TensorIM) -> TensorIM:
        return TensorIM(functional.relu(inputs[0].data + inputs[1].data))


class FPNGetKeys(Module):
    '''将输入的Odict中的键(names)取出为list'''

    def forward(self, odict: ODictIM) -> StringListIM:  # Odict[str, TensorIM]
        names = list(odict.data.keys())
        return StringListIM(names)


class FPNGetValues(Module):
    '''将输入的Odict中的值(x)取出为list'''

    def forward(self, odict: ODictIM) -> TListIM:
        x = list(odict.data.values())
        return TListIM(x)


class GetSubTensor(Module):
    '''通过索引获取TListIM的一个子部分'''
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def forward(self, x: TListIM) -> TensorIM:
        assert isinstance(x.data[self.idx], Tensor)
        return TensorIM(x.data[self.idx])


class FPNResultMerge(MergeModule):
    '''把图像金字塔的多层输出合并为一个List[Tensor]'''

    def forward(self, *inputs: TensorIM) -> TListIM:
        result = []
        for i in inputs:
            assert isinstance(i, TensorIM)
            result.insert(0, i.data)  # 将result排成倒序，先产生的result需要排在后面
        return TListIM(result)


class FPNAdd(MergeModule):
    '''将两个TensorIM相加'''

    def forward(self, t1: TensorIM, t2: TensorIM) -> TensorIM:
        return t1 + t2


class MergeToTList(MergeModule):
    '''RPNHead, BoxPredictor中，将多个TensorIM输出合并为一个List[Tensor]'''

    def forward(self, *inputs: TensorIM) -> TListIM:
        result = []
        for i in inputs:
            assert isinstance(i, TensorIM)
            result.append(i.data)
        return TListIM(result)


class MergeToTupleTList(MergeModule):
    '''RPNHead中多个TListIM合并成TupleTListIM 即Tuple[List[Tensor]]'''

    def forward(self, *inputs: TListIM) -> TupleTListIM:
        data = tuple(ipt.data for ipt in inputs)
        return TupleTListIM(data)


class TupleFork(ForkModule):
    '''接受任意个输入，将它们组合为TupleIM输出'''

    def forward(self, *inputs: Any) -> TupleIM:
        return TupleIM(inputs)


class RPNGetImages(ForkModule):
    '''从(images: ImageListIM, features: OdictIM)元组获取images'''

    def forward(self, tuple_: TupleIM) -> ImageListIM:
        return tuple_.data[0]


class RPNGetFeatures(ForkModule):
    '''从(images: ImageListIM, features: OdictIM)元组获取features'''

    def forward(self, tuple_: TupleIM) -> TListIM:
        features = list(tuple_.data[1].data.values())
        return TListIM(features)


class GetElem(Module):
    '''接受一个TupleIM，获取它的第idx个元素'''
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def forward(self, lst: TupleIM) -> IMData:
        return lst.data[self.idx]


class RoIFlatten(Module):
    '''RoI中，对Tensor执行flatten操作'''

    def forward(self, x: TensorIM) -> TensorIM:
        x = x.data.flatten(start_dim=1)
        return TensorIM(x)


class GetImageShapes(Module):
    '''RoI中，从TupleIM(Tuple(images, features, proposals))里获取image_shapes'''

    def forward(self, tuple_: TupleIM) -> ImageShapeIM:
        return ImageShapeIM(tuple_.data[0].data.image_sizes)

'''
各模块的Rules
'''


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


class BackBoneRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, BackBoneWrap)  # 这里直接判断是否为Wrap模型

    @staticmethod
    def build_dag(block: BackBoneWrap) -> List[RawLayer]:
        img2tensor = RawLayer(0, ImgList2Tensor(), 'img2tensor', [], [])
        body = RawLayer(1, BodyWrap(block.backbone.body), 'body', [], [img2tensor])  # 这里用.backbone跳过了这层展开
        fpn = RawLayer(2, FPNWrap(block.backbone.fpn), 'fpn', [], [body])
        img2tensor.ds_layers = [body]
        body.ds_layers = [fpn]
        return [img2tensor, body, fpn]


class BodyRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, BodyWrap)  # models._utils.IntermediateLayerGetter(ModuleDict)

    @staticmethod
    def build_dag(block: BodyWrap) -> List[RawLayer]:
        '''body的计算包括一系列顺序运行的模型 + 将需要其中返回的层(return_layers)的输出merge为OrderedDict'''
        body = block.body
        # 将OrderedDict分解为两个list，方便处理
        names = list(body.keys())
        modules = list(body.values())
        seq_layers = []
        for idx, module in enumerate(modules):  # 统一添加TensorWrap
            seq_layers.append(RawLayer(idx, ConvertWrap(module), names[idx], [], []))
        for i in range(len(seq_layers)):  # 添加前驱后继
            if i != 0:
                seq_layers[i].ac_layers = [seq_layers[i - 1]]
            if i != len(seq_layers) - 1:
                seq_layers[i].ds_layers = [seq_layers[i + 1]]
        # 处理merge部分
        out_layers_idx = [i for i in range(len(names)) if names[i] in body.return_layers]  # 找到需要作为输出的layers的索引
        out_names = [body.return_layers[names[i]] for i in out_layers_idx]  # 应类似 0,1,2,3,...
        merge_layer = RawLayer(len(seq_layers), MergeFeatures(out_names), 'MergeFeatures',
                               [], [seq_layers[i] for i in out_layers_idx])
        for i in out_layers_idx:
            (seq_layers[i].ds_layers).append(merge_layer)
        return seq_layers + [merge_layer]


class BottleneckRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, Bottleneck)

    @staticmethod
    def build_dag(block: Bottleneck) -> List[RawLayer]:
        ipt = RawLayer(0, ConvertWrap(BasicFork()), 'ipt', [])
        conv1 = RawLayer(1, ConvertWrap(block.conv1), 'conv1', [], [ipt])
        bn1 = RawLayer(2, ConvertWrap(block.bn1), 'bn1', [], [conv1])
        relu1 = RawLayer(3, ConvertWrap(nn.ReLU(inplace=False)), 'relu1', [], [bn1])
        conv2 = RawLayer(4, ConvertWrap(block.conv2), 'conv2', [], [relu1])
        bn2 = RawLayer(5, ConvertWrap(block.bn2), 'bn2', [], [conv2])
        relu2 = RawLayer(6, ConvertWrap(nn.ReLU(inplace=False)), 'relu2', [], [bn2])
        conv3 = RawLayer(7, ConvertWrap(block.conv3), 'conv3', [], [relu2])
        bn3 = RawLayer(8, ConvertWrap(block.bn3), 'bn3', [], [conv3])
        ipt.ds_layers = [conv1]
        conv1.ds_layers = [bn1]
        bn1.ds_layers = [relu1]
        relu1.ds_layers = [conv2]
        conv2.ds_layers = [bn2]
        bn2.ds_layers = [relu2]
        relu2.ds_layers = [conv3]
        conv3.ds_layers = [bn3]
        if block.downsample is not None:  # 连接ipt的后继和merge的前驱时，要注意两者的分支顺序相同
            downsample0 = RawLayer(9, ConvertWrap(block.downsample[0]), 'downsample0', [], [ipt])
            downsample1 = RawLayer(10, ConvertWrap(block.downsample[1]), 'downsample1', [], [downsample0])
            merge = RawLayer(11, BottleneckAdd(), 'ba', [], [bn3, downsample1])
            ipt.ds_layers.append(downsample0)
            downsample0.ds_layers = [downsample1]
            downsample1.ds_layers = [merge]
            bn3.ds_layers = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, downsample0, downsample1, merge]
        else:
            merge = RawLayer(9, BottleneckAdd(), 'ba', [], [bn3, ipt])
            ipt.ds_layers.append(merge)
            bn3.ds_layers = [merge]
            return [ipt, conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, merge]


class FPNRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, FPNWrap)  # models.detection.ops.FeaturePyramidNetwork

    @staticmethod
    def build_dag(block: FPNWrap) -> List[RawLayer]:
        id_mng = IDManager()

        fpn = block.fpn
        inner_blocks = [ConvertWrap(module) for module in fpn.inner_blocks]
        layer_blocks = [ConvertWrap(module) for module in fpn.layer_blocks]

        ipt = RawLayer(id_mng.new_id(), BasicFork(), 'ipt', [], [])
        get_names = RawLayer(id_mng.new_id(), FPNGetKeys(), 'getkey', [], [ipt])
        get_x = RawLayer(id_mng.new_id(), FPNGetValues(), 'getval', [], [ipt])
        last_sub_ts = RawLayer(id_mng.new_id(), GetSubTensor(-1), 'subts[-1]', [], [get_x])
        last_inner = RawLayer(id_mng.new_id(), inner_blocks[-1], 'inner[-1]', [], [last_sub_ts])
        last_layer = RawLayer(id_mng.new_id(), layer_blocks[-1], 'layer[-1]', [], [last_inner])
        res_modules = [last_layer]  # 所有需要merge的结果

        x_len = len(inner_blocks)  # 用inner_blocks的长度替代获取x的长度。建立backbone时，len(return layers)决定了len(fpn.inner_blocks)和len(body部分输出).
        for idx in range(x_len - 2, -1, -1):
            sub_ts = RawLayer(id_mng.new_id(), GetSubTensor(idx), f'subts[{idx}]', [], [get_x])
            inner_lateral = RawLayer(id_mng.new_id(), inner_blocks[idx], f'inner[{idx}]', [], [sub_ts])
            interpolate = RawLayer(id_mng.new_id(), InterpolateWrap(), 'interp', [], [last_inner, inner_lateral])
            last_inner = RawLayer(id_mng.new_id(), FPNAdd(), 'add', [], [inner_lateral, interpolate])
            layer_lateral = RawLayer(id_mng.new_id(), layer_blocks[idx], f'layer[{idx}]', [], [last_inner])
            res_modules.append(layer_lateral)

        merge_res = RawLayer(id_mng.new_id(), FPNResultMerge(), 'merge', [], res_modules)
        extra_block = RawLayer(id_mng.new_id(), ExtraBlockWrap(fpn.extra_blocks), 'extra', [], [get_names, get_x, merge_res])
        layers = collect_all_layers(extra_block)
        fill_ds_layers(layers)
        return layers

class RPNHeadRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.rpn.RPNHead)

    @staticmethod
    def build_dag(head: models.detection.rpn.RPNHead) -> List[RawLayer]:
        id_mng = IDManager()
        ipt = RawLayer(id_mng.new_id(), BasicFork(), 'ipt', [], [])
        cls_layers = []
        bbox_layers = []
        for idx in range(5): # len(feature) == 5
            sub_ts = RawLayer(id_mng.new_id(), GetSubTensor(idx), f'subts[{idx}]', [], [ipt])
            conv = RawLayer(id_mng.new_id(), ConvertWrap(head.conv), 'conv', [], [sub_ts])
            relu = RawLayer(id_mng.new_id(), ConvertWrap(nn.ReLU()), 'relu', [], [conv])
            cls_logits = RawLayer(id_mng.new_id(), ConvertWrap(head.cls_logits), 'cls_conv', [], [relu])
            bbox_pred = RawLayer(id_mng.new_id(), ConvertWrap(head.bbox_pred), 'bbox_conv', [], [relu])
            cls_layers.append(cls_logits)
            bbox_layers.append(bbox_pred)
        cls = RawLayer(id_mng.new_id(), MergeToTList(), 'cls_merge', [], cls_layers)
        bbox = RawLayer(id_mng.new_id(), MergeToTList(), 'bbox_merge', [], bbox_layers)
        merge = RawLayer(id_mng.new_id(), MergeToTupleTList(), 'merge', [], [cls, bbox])
        layers = collect_all_layers(merge)
        fill_ds_layers(layers)
        return layers


class RPNRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, RegionProposalNetwork)

    @staticmethod
    def build_dag(rpn: RegionProposalNetwork) -> List[RawLayer]:
        id_mng = IDManager()
        ipt = RawLayer(id_mng.new_id(), TupleFork(), 'ipt', [], [])
        images = RawLayer(id_mng.new_id(), RPNGetImages(), 'images', [], [ipt])
        features = RawLayer(id_mng.new_id(), RPNGetFeatures(), 'features', [], [ipt])
        rpn_head = RawLayer(id_mng.new_id(), rpn.head, 'rpn_head', [], [features])
        anchors = RawLayer(id_mng.new_id(), RPNAnchorGenWrap(rpn.anchor_generator), 'anchor_gen', [], [images, features])
        other_cal = RawLayer(id_mng.new_id(), RPNCalculateWrap(rpn.box_coder, rpn.filter_proposals), 'other_cal', [], [images, rpn_head, anchors])
        layers = collect_all_layers(other_cal)
        fill_ds_layers(layers)
        return layers


class TwoMLPHeadRule(BlockRule):
    '''对应RoI中的box_head'''
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.faster_rcnn.TwoMLPHead)

    @staticmethod
    def build_dag(block: models.detection.faster_rcnn.TwoMLPHead) -> List[RawLayer]:
        id_mng = IDManager()
        flatten = RawLayer(id_mng.new_id(), RoIFlatten(), 'flatten', [], [])
        fc6 = RawLayer(id_mng.new_id(), ConvertWrap(block.fc6), 'fc6', [], [flatten])
        relu1 = RawLayer(id_mng.new_id(), ConvertWrap(nn.ReLU()), 'relu',[], [fc6])
        fc7 = RawLayer(id_mng.new_id(), ConvertWrap(block.fc7), 'fc7', [], [relu1])
        relu2 = RawLayer(id_mng.new_id(), ConvertWrap(nn.ReLU()), 'relu', [], [fc7])
        layers = collect_all_layers(relu2)
        fill_ds_layers(layers)
        return layers


class FastRCNNPredictorRule(BlockRule):
    '''对应RoI中的box_predictor'''
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.faster_rcnn.FastRCNNPredictor)

    @staticmethod
    def build_dag(block: models.detection.faster_rcnn.FastRCNNPredictor) -> List[RawLayer]:
        id_mng = IDManager()
        flatten = RawLayer(id_mng.new_id(), RoIFlatten(), 'flatten', [], [])
        scores = RawLayer(id_mng.new_id(), ConvertWrap(block.cls_score), 'cls_score', [], [flatten])
        bbox_deltas = RawLayer(id_mng.new_id(), ConvertWrap(block.bbox_pred), 'bbox_pred', [], [flatten])
        merge = RawLayer(id_mng.new_id(), MergeToTList(), 'merge', [], [scores, bbox_deltas])
        layers = collect_all_layers(merge)
        fill_ds_layers(layers)
        return layers


class RoIRule(BlockRule):
    @staticmethod
    def is_target(module: Module) -> bool:
        return isinstance(module, models.detection.roi_heads.RoIHeads)

    @staticmethod
    def build_dag(block: models.detection.roi_heads.RoIHeads) -> List[RawLayer]: # [images: ImageListIM, features: OdictIM, proposals: TListIM] -> DetectionIM
        id_mng = IDManager()
        ipt = RawLayer(id_mng.new_id(), TupleFork(), 'ipt', [], [])
        image_shapes = RawLayer(id_mng.new_id(), GetImageShapes(), 'images', [], [ipt])
        proposals = RawLayer(id_mng.new_id(), GetElem(2), 'props', [], [ipt])
        box_roi_pool = RawLayer(id_mng.new_id(), BoxRoIPoolWrap(block.box_roi_pool), 'box_roi_pool', [], [ipt])
        box_head = RawLayer(id_mng.new_id(), block.box_head, 'box_head', [], [box_roi_pool])
        box_pred = RawLayer(id_mng.new_id(), block.box_predictor, 'box_pred', [], [box_head])
        other_cal = RawLayer(id_mng.new_id(), RoICalculateWrap(block.postprocess_detections), 'other_cal', [], [image_shapes, proposals, box_pred])
        layers = collect_all_layers(other_cal)
        fill_ds_layers(layers)
        return layers

'''
utils
'''
class IDManager:
    def __init__(self):
        self._id = -1

    def new_id(self):
        self._id += 1
        return self._id


def collect_all_layers(last_layer: RawLayer) -> List[RawLayer]:
    '''从最后一层开始向前遍历，返回所有layer按id排序的List'''
    queue = [last_layer]
    seen = set(queue)
    while any(queue):
        ly = queue.pop()
        for ac in ly.ac_layers:
            if ac not in seen:
                seen.add(ac)
                queue.append(ac)
    layers = list(seen)
    assert len(layers) == last_layer.id_ + 1
    return sorted(layers, key=lambda x: x.id_)


def fill_ds_layers(layers: List[RawLayer]):
    '''根据前驱，为每个节点填写以id排序好的后继节点'''
    for idx in range(len(layers)): # 遍历每个节点
        for ac in layers[idx].ac_layers: # 遍历其所有前驱
            layers[ac.id_].ds_layers.append(layers[idx]) # 为前驱添加该节点为其后继
    for idx in range(len(layers)):
        layers[idx].ds_layers.sort(key=lambda x: x.id_) # 保证后继有序


def prepare_fasterrcnn() -> DNNConfig:
    frcnn = models.detection.fasterrcnn_resnet50_fpn(True)
    frcnn.eval()
    return DNNConfig('frcnn', frcnn, {RCNNRule, BackBoneRule, BodyRule, BottleneckRule, FPNRule, \
                                      RPNHeadRule, RPNRule, TwoMLPHeadRule, FastRCNNPredictorRule, RoIRule}, {})


if __name__ == '__main__':
    raw_dnn = RawDNN(prepare_fasterrcnn())
    for ly in raw_dnn.layers:
        print(ly)
