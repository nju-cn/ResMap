# 开发笔记

## 2021.11.10

- [x] 添加了Worker，带Scheduler的Master以及相应gRPC的Servicer，Master和1个Worker测试误差在3e-6到5e-6左右，但这个误差是Master在验证计算时带来的，不是Worker带来的
  - [x] config.yml：配置文件
  - [x] logging.yml：日志配置文件
  - [x] msg.proto：添加了Master服务和相关结构体，IFR的JobMsg改成WkJobMsg
  - [x] main.py：启动入口
  - [x] scheduler.py：只为AlexNet的一个worker生成IFR
  - [x] master.py：产生并发送IFR任务
  - [x] master_servicer.py：使用grpc提供服务
  - [x] worker.py：定义IFR，获取并执行Job
  - [x] worker_servicer.py：使用grpc提供服务

## 2021.11.7

- [x] 修正了是否用CSR的判断依据，单元测试正常
  - [x] dif_executor.py：之前按照1/3判断是错的，更新了公式；应该用CSC时输出warning
  - [x] unit_tests/test_dif_executor.py：测试数据大小使用360p，非零占比为49.5%

## 2021.11.5

- [x] 添加了对DifJob的序列化，单元测试正常
  - [x] msg.proto：序列化数据格式
  - [x] msg_compile.cmd：编译proto文件
  - [x] dif_executor.py：Tensor、DifJob和pb数据转换
  - [x] unit_tests/test_dif_executor.py：添加了序列化数据转换的单元测试
  - [x] .gitignore：屏蔽了pb生成的代码

## 2021.11.3

- [x] 使用Git LFS管理视频数据
  - [x] .gitattributes：Git LFS配置文件
  - [x] media：改用Git LFS管理

## 2021.11.2

- [x] 修复了make_dag打印日志的bug和IntegralExecutor的bug，把视频放到了media目录下。DNN生成信息不再重复打印；dif_executor的fixed_jobs和var_jobs测试平均误差都在1e-6以内。
  - [x] dnn_dag.py：修复了默认logger重复打印DAG生成信息的问题
  - [x] integral_executor.py：IntegralExecutor执行完一个job要重置所有node，不能只重置job指定的node
  - [x] dif_executor.py：新增了单元测试var_jobs
  - [x] media：添加了林荫道路视频road.mp4

- [x] 把IntegralExecutor和DifExecutor的测试代码放到了单独的unit_tests目录下。测试全部通过
  - [x] integral_executor.py：去掉了在main中的测试代码
  - [x] dif_executor.py：去掉了在main中的测试代码
  - [x] unit_tests：单元测试的目录
    - [x] common.py：单元测试共用的函数，如从视频中读取一帧
    - [x] test_integral_executor.py：对IntegralExecutor的单元测试，包括各类CNN
    - [x] test_dif_executor.py：对DifExecutor的单元测试，使用差值数据，运行中worker的job会变化

## 2021.11.1

- [x] Executor作为接口，使用IntegralExecutor和DifExecutor分别执行完整的Job（IntegralJob）和仅有差值的Job（DifJob）。integral_executor单元测试误差为0，dif_executor单元测试误差在1e-6以内
  - [x] executor.py：接口定义
  - [x] dnn_dag.py：make_dag的logger参数添加了默认值
  - [x] integral_executor.py：把单元测试放进函数，RawLayer执行放入IntegralExecutor
  - [x] dif_executor.py：使用InCache和OutCache保存上一帧数据，测试了DifJob固定时执行的误差；DifJob和rpc之间的转换还没测试，所以先注释掉

## 2021.10.31

- [x] Executor添加了内存回收，使得内存占用最小，executor运行ResNet50正常
  - [x] executor.py：单元测试代码检查output非None的节点，结果正常
- [x] 去掉了单元测试代码，executor运行ResNet50正常
  - [x] executor.py：优化代码风格

## 2021.10.30

- [x] 使用了Executor，IFR，Job来对Worker执行的任务进行抽象，executor运行AlexNet正常
  - [x] chain.py：添加了prepare_vgg16
  - [x] googlenet.py：InceptionCat把列表传参改成了用`*inputs`传参数
  - [x] resnet.py：BottleneckAdd把列表传参改成了用`*inputs`传参数
  - [x] dnn_dag.py：`make_dag`生成RawLayer后，把所有的ReLU.inplace置为False
  - [x] node.py：calc直接使用RawLayer中的calc；去除对padding的修改；去除对ReLU.inplace的修改；添加了DeepSlicing中的Master.init_dag_range，并使用init_rdag进行封装，用于初始化各Node的输入输出区间
  - [x] executor.py：Worker一次执行多个层为一个Job，一个帧对应的执行计划为一个IFR，ExNode用于保存单个层的输出数据，Executor对于一个CNN的任意Job进行执行

- [x] executor运行VGG19正常
- [x] 修复了RawLayer运行的一个小问题，executor以链状Job的方式运行GoogLeNet正常
  - [x] googlenet.py：打印出所有layer
  - [x] dnn_dag.py：修复了execute_dag中有多个输入时报错的问题
  - [x] executor.py：改用了GoogLeNet，多个Job只有一条边相连
- [x] executor以子图切割的方式运行GoogLeNet正常
  - [x] executor.py：GoogLeNet的Job之间有多条边相连
  - [x] echarts_util.py：把echarts图变大了，方便查看
- [x] executor以子图切割的方式运行ResNet50正常
  - [x] executor.py：ResNet50的Job之间有多条边相连
  - [x] resnet.py：打印出所有layer，代码风格优化
  - [x] googlenet.py：代码风格优化