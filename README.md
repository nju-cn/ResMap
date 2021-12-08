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

### 不同局域网下的内网穿透

但是如果设备在不同的局域网下时，需要确保上述表格中设备的互相可访问。这可以通过[frp](https://github.com/fatedier/frp)内网穿透实现。注意树莓派需要下载arm版（不是arm64）。

首先准备一台公网服务器，假设IP地址为`121.199.15.155`，把该IP地址的端口分别映射到各个设备，如config.yml中的部分配置如下：

```yaml
addr:
  master: '121.199.15.155:11100'
  worker:
    0: '121.199.15.155:11110'
    1: '121.199.15.155:11111'
    2: '121.199.15.155:11112'
  trainer: '121.199.15.155:11120'
```

然后配置内网穿透。

公网服务器上运行一个frps，配置frps.ini如下：

```ini
[common]
bind_port = 7000
# http://121.199.15.155:7500 可以查看面板
dashboard_port = 7500
```

pi4B-2g1上运行了Master，所以它运行一个frpc，配置frpc.ini如下：

```ini
[common]
# 121.199.15.155为公网上frps的IP地址
server_addr = 121.199.15.155
server_port = 7000

# Master服务
[master]
type = tcp
# 远程映射端口，其他设备通过121.199.15.155:11100访问
remote_port = 11100
# Master运行在本机上，所以就是127.0.0.1
local_ip = 127.0.0.1
# 本地运行端口，必须和远程映射端口相同，本机上监听的是11100
local_port = 11100
```

类似地，pi4G上运行了Worker0，所以它也运行一个frpc，配置frpc.ini如下：

```ini
[common]
server_addr = 121.199.15.155
server_port = 7000

[worker0]
type = tcp
remote_port = 11110
local_ip = 127.0.0.1
local_port = 11110
```

类似地，PC上运行了Trainer和Worker1，所以它也运行一个frpc，配置frpc.ini如下：

```ini
[common]
server_addr = 121.199.15.155
server_port = 7000

[trainer]
type = tcp
remote_port = 11120
local_ip = 127.0.0.1
local_port = 11120

[worker1]
type = tcp
remote_port = 11111
local_ip = 127.0.0.1
local_port = 11111
```

在公网服务器上启动frps：

```bash
./frps -c frps.ini
```

在各台设备上启动frpc：

```bash
./frpc -c frpc.ini
```

## 运行实验

网络就绪后，就可以运行实验了。

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

