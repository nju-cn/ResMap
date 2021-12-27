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

### 拓扑结构

要确保如下的设备访问路径可行，即client可以通过特定的IP端口号访问到相应的server。

| 设备(client) | 要访问的设备(server) |
| ------------ | -------------------- |
| Master       | 所有Worker，Trainer  |
| Worker i     | Worker i+1           |

虽然Worker也会作为client请求Master，但是系统内部通过grpc的流式回复实现了这一功能，所以无需配置。

### 网络限速

使用 [cgroup](https://man7.org/linux/man-pages/man7/cgroups.7.html) 和 [tc](https://man7.org/linux/man-pages/man8/tc.8.html) 对指定进程进行限速。基本原理就是，tc对指定cgroup的所有进程限制总带宽为一个特定值。这里我们只需要对一个进程限速，所以cgroup中只会有一个进程。

#### 安装

tc系统自带，只需安装cgroup工具，以便使用cgexec命令。

```bash
sudo apt install cgroup-tools
```

#### 创建cgroup和tc规则

```bash
# 在net_cls下创建名为mylim的cgroup
cd /sys/fs/cgroup/net_cls/
sudo su
mkdir mylim
cd mylim/
echo 0x10010 > net_cls.classid
# 创建对于wlan0网卡的root规则，qdisc编号为1
tc qdisc add dev wlan0 root handle 1: htb
# 添加过滤器，使用cgroup号作为class
tc filter add dev wlan0 parent 1: handle 1: cgroup
# 添加class，cgroup号为1:10的进程网速限制为1MB/s
tc class add dev wlan0 parent 1: classid 1:10 htb rate 1mbps
```

#### 把服务运行在指定cgroup中

以启动worker0为例，命令如下

```bash
sudo -E PATH=$PATH PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))") cgexec -g net_cls:mylim /usr/bin/python3.7 main.py worker -i 0
```

`cgexec -g net_cls:mylim` 指定后面的命令启动的进程运行在net_cls下的mylim这个cgroup中

cgexec需要sudo权限，但是因为Python的包都是安装在当前用户下而不是系统安装，所以root用户目录下运行python会找不到很多包。所以这里要设置PATH和PYTHONPATH（就是`sys.path`）告诉解释器从哪里加载包。此外，sudo的E选项表示运行后面命令时保留当前的环境变量。

## 运行实验

网络就绪后，就可以运行实验了。

首先在rpc目录下运行脚本msg_compile.cmd，生成grpc文件。

然后在config.yml中填写相关配置。

启动Trainer

```bash
python3 main.py trainer
```

启动各个Worker，如Worker0。如果需要限速则按照前述方法启动。

```bash
python3 main.py worker -i 0
```

最后启动Master（注意Master要在Worker和Trainer之后启动）。如果需要限速则按照前述方法启动。

```bash
python3 main.py master
```

