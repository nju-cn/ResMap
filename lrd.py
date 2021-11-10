from math import ceil, floor
from typing import Callable, Tuple, Optional

from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, Module, ReLU

from dnn_config import InputModule, BasicFork


class NoOutException(Exception):
    """给定输入无法产生任何输出时抛出的异常"""
    def __init__(self, calc: Module, i_begin: int, i_end: int, o_begin: int, o_end: int):
        self.calc = calc
        self.i_begin = i_begin
        self.i_end = i_end
        self.o_begin = o_begin
        self.o_end = o_end

    def __str__(self):
        return f"{self.calc}的out_range({self.i_begin},{self.i_end})结果为[{self.o_begin},{self.o_end}]," \
            f" {self.o_begin}>{self.o_end}！"


class _OutRangeFactory:
    """使用此类解决out_range_factory生成的函数不支持pickle的问题，不支持外部引用"""

    def __init__(self, calc: Optional[Module] = None):
        self.__calc = calc

    def out_range(self, x1: int, x2: int, idx: int, strict: bool = False) -> Tuple[int, int]:
        """根据传入的calc实现out_range
        这里只考虑所需要被切割维度的数据区间，不保证有[x1, x2]的数据就可以算出[返回值0, 返回值1]
        如MergeModule需要所有前驱分支的此区间数据，ceil_mode=True的Pooling层可以在边缘缺失的情况下给出输出
        严格模式(strict=True)的out_range与req_range是互为逆操作的：out_range(x1,x2)=(y1,y2)且req(y1,y2)=(x1,x2)
        注意：strict=False不是宽松模式，而是与calc保持一致，这里和req_range不同
             strict=False的计算会把[x1,x2]当做整个输入区间，
             若[x1,x2]只是输入的一部分且不位于右侧边界上，则必须strict=True，否则会产生错误的结果
        """
        assert x1 <= x2, f"out_range的参数{x1}>{x2}！"
        calc = self.__calc
        if calc is None:
            return x1, x2
        # 根据calc函数确定输出范围out_range
        # 计算[x1, x2]的输出范围所需要的输入范围[返回值0, 返回值1]，全部闭区间
        # 注意：dilation没有考虑
        if isinstance(calc, Conv2d) or isinstance(calc, MaxPool2d):
            if isinstance(calc, MaxPool2d) and calc.ceil_mode:
                end_round = ceil
            else:
                end_round = floor
            # 此公式是由conv2d的req_range公式反推出来的
            if strict:  # strict=True，无论calc的ceil_mode是什么，都按照ceil_mode=False，用于位于中间部分的Slice数据
                # 严格模式，严格按照现有数据计算，边缘处的数据不够就没有相应输出。此时与req_range可以保持一致。
                out_round = floor
            else:  # strict=False，默认值，与calc保持一致，用于位于输入数据边缘部分的Slice数据
                # calc的ceil_mode=True时对存在缺失的边缘部分数据也可以产生输出，ceil_mode=False时则不行(此时与strict=True一样)
                out_round = end_round
            K = (calc.kernel_size if isinstance(calc.kernel_size, int) else calc.kernel_size[idx])
            S = (calc.stride if isinstance(calc.stride, int) else calc.stride[idx])
            # 输入必须加padding，因为不知道计算数据整体区间，所以此函数中不知道要把padding加在哪里
            # 因为[x1,x2]就是整体区间和只是在整体区间内时输出是不一样的。整体区间两边都要pad，而只是在内部则两边都不需要pad
            y1, y2 = ceil(x1 / S), out_round((x2 - K + 1) / S)
            # 对于ceil_mode=True的MaxPool2d，y1>y2的情况是可能出现的，如x1=x2且K=3
            if y1 > y2:
                raise NoOutException(calc, x1, x2, y1, y2)
            return y1, y2
        elif isinstance(calc, BatchNorm2d) or isinstance(calc, InputModule) \
                or isinstance(calc, BasicFork) or isinstance(calc, ReLU):
            return x1, x2
        else:
            raise AssertionError(f"New Node Type: {type(calc)}")


def out_range_factory(calc: Optional[Module] = None) -> Callable[[int, int, int, bool], Tuple[int, int]]:
    """根据calc函数确定输出范围out_range，不传入calc则直接返回默认的直接映射(x1,x2)=>(x1,x2)
    out_range：输入范围[x1, x2]所对应的输出范围[返回值0, 返回值1]，全部闭区间
    """
    return _OutRangeFactory(calc).out_range


class _ReqRangeFactory:
    """使用此类解决req_range_factory生成的函数不支持pickle的问题，不支持外部引用"""

    def __init__(self, calc: Optional[Module] = None):
        self.__calc = calc

    def req_range(self, x1: int, x2: int, idx: int, strict: bool = True) -> Tuple[int, int]:
        """根据传入的calc实现req_range
        这里不需要考虑MaxPool2d的ceil_mode但是需要注意req_range返回值可能会超出输入图像的范围(ceil_mode=True)
        这里只考虑所需要被切割维度的数据区间，不保证有此区间的数据就可以进行计算，如MergeModule需要所有前驱分支的此区间数据
        默认strict=True为严格模式，此时与严格模式(strict=True)的out_range是互为逆操作的
        strict=False时为宽松模式，会返回可以计算出[x1,x2]的最小输入范围，前提是相应calc的ceil_mode=True否则报错
        """
        assert x1 <= x2, f"req_range的参数{x1}>{x2}！"
        calc = self.__calc
        if not strict:
            assert isinstance(calc, MaxPool2d) and calc.ceil_mode, f"{calc}不应该使用宽松模式！"
        if calc is None:
            return x1, x2
        # 注意：dilation没有考虑
        if isinstance(calc, Conv2d) or isinstance(calc, MaxPool2d):
            # idx表示维度(行还是列),0为行,1为列
            # req_range总是严格模式，因为ceil_mode=True的情况下[x1,x2]相应的输入区间不是唯一的
            K = (calc.kernel_size if isinstance(calc.kernel_size, int) else calc.kernel_size[idx])
            S = (calc.stride if isinstance(calc.stride, int) else calc.stride[idx])
            # 返回的结果只能包括padding，因为不知道整体计算区间，所以无法得到pad前的原始输入区间
            if strict:
                y1, y2 = x1 * S, x2 * S + K - 1
            else:  # 返回可以计算出的最小范围
                y1, y2 = x1 * S, (x2-1) * S + K
            # 防止出现y1>y2的情况
            assert y1 <= y2, f"{calc}的req_range({x1},{x2})结果为[{y1},{y2}], {y1}>{y2}！"
            return y1, y2
        elif isinstance(calc, BatchNorm2d) or isinstance(calc, InputModule) \
                or isinstance(calc, BasicFork) or isinstance(calc, ReLU):
            return x1, x2
        else:
            raise AssertionError(f"New Node Type: {type(calc)}")


def req_range_factory(calc: Optional[Module] = None) -> Callable[[int, int, int, bool], Tuple[int, int]]:
    """根据calc函数确定需要的输入范围req_range，不传入calc则直接返回默认的直接映射(x1,x2)=>(x1,x2)
    req_range：计算[x1, x2]的输出范围所需要的输入范围[返回值0, 返回值1]，全部闭区间
    """
    return _ReqRangeFactory(calc).req_range


if __name__ == '__main__':
    import torchvision.models as models
    from torchvision.transforms import transforms
    from torch.nn import Sequential
    from PIL import Image

    googlenet = models.googlenet(True)
    googlenet.eval()
    before = Sequential(
        googlenet.conv1,
        googlenet.maxpool1,
        googlenet.conv2,
        googlenet.conv3
    )

    input_image = Image.open("bird.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(451),  # 变为301*301
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    begin, end = 0, 450
    for bf in before:
        for calc in bf.children():
            if isinstance(calc, Conv2d):
                begin, end = begin, end + calc.padding[0] * 2
            begin, end = out_range_factory(calc)(begin, end, 0)
            print("calc:", begin, end)
            input_batch = calc(input_batch)
            print("fact:", input_batch.shape)
        if len(list(bf.children())) == 0:
            end += (bf.padding if isinstance(bf.padding, int) else bf.padding[0])
            begin, end = out_range_factory(bf)(begin, end, 0)
            print("calc:", begin, end)
            input_batch = bf(input_batch)
            print("fact:", input_batch.shape)
        print("-----")
