# 使用说明

## 环境配置

Python版本需要3.7及以上。

### Ubuntu

安装Python依赖包和系统的OpenCV库

```bash
pip3 install -r requirements.txt
sudo apt update
sudo apt install python3-opencv
```

### 树莓派

用如下命令安装Python依赖包，除torch和torchvision外都会安装成功。

```bash
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
```

下载安装torch和torchvision的wheel

```bash
wget https://github.com/zsnjuts/pytorch-armv7l/archive/refs/tags/v1.0.zip
unzip v1.0.zip
pip3 install pytorch-armv7l-1.0/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
pip3 install pytorch-armv7l-1.0/torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
rm -rf pytorch-armv7l-1.0/ v1.0.zip
```

安装系统的依赖库

```bash
sudo apt update
sudo apt install libatlas-base-dev libopenblas-dev python3-opencv python3-numpy python3-scipy
```

## 网络配置

设备互相请求如下：

| 设备     | 要访问的设备        |
| -------- | ------------------- |
| Master   | 所有Worker，Trainer |
| Worker i | Worker i+1          |
| Trainer  | 无                  |

如果所有设备全部在同一个局域网下，则没有任何问题。配置好IP地址即可运行。

但是如果设备在不同的局域网下时，需要确保上述表格中设备的互相可访问。

## 运行实验

在config.yml中填写相关配置

启动Trainer

```bash
python3 main.py trainer
```

启动各个Worker，如Worker0

```bash
python3 main.py worker -i 0
```

最后启动Master（注意Master要在Worker和Trainer之后启动）

```bash
python3 main.py master
```

