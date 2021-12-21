# 开发笔记

## 2021.12.21

- [x] :four_leaf_clover: [实验] 添加了lcnz_show.py，以便展示每层输出数据差值不同通道的非零占比
- [x] :four_leaf_clover: [实验] 修改了ionz_show.py，使之可以在原图上显示预测结果 

## 2021.12.20

- [x] Worker的LayerCost缓存文件名包含了帧大小，log做了小调整。PC本地协同测试正常
- [x] :four_leaf_clover: [实验] 添加了costs_show.py，以便展示worker的计算耗时
- [x] :four_leaf_clover: [实验] 添加了lnz_show.py，以便展示各层输出差值的稀疏情况
- [x] :four_leaf_clover: [实验] 添加了ionz_show.py，以便展示每层的输入差值非零占比和输出差值非零占比的关系
- [x] :four_leaf_clover: [实验] 添加了img_dif_show.py，以便对于两张先后的图像，展示它们在CNN各层输出数据差值的非零占比变化曲线
- [x] :four_leaf_clover: [实验] 修改了lfcnz_gen.py，直接调用Trainer相关函数生成LFCNZ数据
- [x] :four_leaf_clover: [实验] 补交img_dif_show所需的两张图像，ionz_show功能小修改

## 2021.12.15

- [x] Scheduler考虑了首个IFR和后续IFR数据传输量不同的问题，Master小修改。pipeline测试正常，PC本地协同测试正常，但是调度策略正确性有待验证
  - [x] master.py：当误差过大时输出warning
- [x] 把Scheduler改成了接口，原先的调度逻辑变成NBCScheduler。nbc_show测试正常，PC本地协同测试正常
- [x] 添加了baseline的调度器LBCScheduler。云边协同测试正常
  * 云边协同运行10帧，dif模式下性能对比：LBC每帧平均1.6s，NBC每帧平均1.55s
  * NBC比LBC效果不明显，原因可能是带宽数据不准确，而且没有考虑压缩和解压缩耗时

- [x] 把配置文件传参改成了每个类的config只有它自己的配置，而非全局配置；其他代码优化。PC本地协同测试正常
  - [x] dnn_config.py，dnn_models目录：DNNConfig添加了dnn名称
  - [x] master.py：把Scheduler初始化放到了单独的函数中
  - [x] lbc_scheduler.py：增加了对ItgJob的支持
  - [x] nsc_scheduler.py，nsc_show.py：NBC改名为NSC
  - [x] util.py：删掉了dnn_abbr函数

## 2021.12.14

- [x] bug修复：Scheduler.gen_wk_jobs返回self.__wk_jobs时要更新输入数据。PC本地协同测试误差正常

## 2021.12.12

- [x] 实现了Worker到Master的逆向通信，不再需要内网穿透了；scheduler中修复了Scheduler.optimize_chain的bug。PC本地协同测试正常，云边协同还没测试，README还没改
  - 这样做的主要原因：有可能IFR在到达最后一个Worker之前完成，所以每一个Worker都需要能够单独给Master发送report_finish，所以需要实现无需内网穿透的Master到Worker的通信渠道
  - 这样做的缺点：最后一个Worker到Master的时延无法获取了，需要通过改用时间戳来计时

- [x] Worker1已完成时直接发给Master而不发给Worker2。PC本地协同测试正常
- [x] 修改Master，使之可以接受IFR1在IFR0之前到达。云边协同测试正常
  * 这样做的原因：因为调度策略可能会发生变化，所以先发出的IFR可能后到达

- [x] 修改了README的网络配置部分
- [x] Scheduler改成了只对IFR0和IFR1进行调度，后面的直接使用IFR1的方案。云边协同测试正常
  * 这样做的原因：每次IFR调度变化都会引起缓存失效，从而导致需要传输全部数据，所以调度策略应该尽可能少地变化。IFR0的数据量和后面的显著不同，所以要单独考虑

## 2021.12.11

- [x] pipeline更新成真实数据，测试正常
- [x] 通过配置itg或dif可以选择运行模式。单元测试正常，PC本地协同测试正常
  * 启动时根据itg或dif增加配置项executor和job
  * tensor和msg互转移到了util
  * IntegralJob改名ItgJob，和DifJob都继承自Job，IFR中用Job而非DifJob

## 2021.12.10

- [x] 添加AsyncClient把new_ifr和report_finish这种无需结果的RPC改成了异步发送请求。PC本地协同测试正常
- [x] 修复了Scheduler.optimize_chain中没有考虑worker无layer的bug，config的带宽使用了真实数据，README完善。云边协同测试正常
  * 两种方法网速测试结果如下：（使用wget从python的httpserver下载；代码中的timed_rpc输出）

| 设备                    | wget测得网速          | grpc测得网速     |
| ----------------------- | --------------------- | ---------------- |
| pi2G1 (m) - pi4G (w0)   | 6.29 MB/s             | 5.34 MB/s        |
| pi4G (w0) - PC (w1)     | 4.45 MB/s             | 7.61 MB/s        |
| PC (w1) - aliyun(w2)    | 695 KB/s = 0.679 MB/s | 1.66 MB/s        |
| aliyun (w2) - pi2G1 (m) | 692 KB/s = 0.676 MB/s | 数据量太小, 未知 |

## 2021.12.9

- [x] 改变了config中的网络表示方式，所有Worker直接使用list表示，添加了对序列化和传输的计时，把通用的StubFactory改成了分Master和Worker的两种实现，配置字典统一命名为config。PC本地协同测试正常，但云边协同还没测，README还没更新
  - [x] config.yml，main.py，master.py，master_servicer.py，stub_factory.py，trainer_servicer.py，util.py，worker.py，worker_servicer.py
- [x] 修复了check=false时报错的问题，Stub加上了debug输出。PC本地协同测试正常，但云边协同还没测，README还没更新
  * 因为写涉及的文件意义不大，所以除了需要特殊说明的，之后开发笔记中每个commit不再写涉及的文件

- [x] 云边协同测试正常，更新了README。测试方案如下：

| 设备   | 服务             | 地址                                     |
| ------ | ---------------- | ---------------------------------------- |
| pi2G1  | master           | 121.199.15.155:11110 （内网穿透）        |
| pi4G   | worker0          | 192.168.0.114:33330                      |
| PC     | worker1, trainer | 114.212.84.93:33331, 114.212.84.93:22220 |
| aliyun | worker2          | 121.199.15.155:33332                     |

## 2021.12.8

- [x] README中添加了网络配置说明，config使用了基于frp内网穿透的配置。按照README的配置，通过frp内网穿透，实现了 端(Master-pi2G1)，边(Worker0-pi4G, Worker1-PC)，云(Worker2-阿里云) 的协同，测试正常
  - [x] requirements.txt：协同时会有sklearn的`UserWarning: Trying to unpickle estimator MLPRegressor from version 0.23.2 when using version 1.0.1`。 因为Trainer版本是0.23.2，但是Master版本是1.0.1，所以要把sklearn的版本统一固定在1.0.1

## 2021.12.7

- [x] 添加了requirements.txt和README.md

## 2021.12.6

- [x] Scheduler中把计算各Worker传输和计算耗时的代码独立成plan2costs_chain，pipeline的可视化函数visualize_frames移到Scheduler。pipeline测试正常(可以看到迭代优化的可视化过程)，但协同测试会崩溃(因为matplotlib只能在主线程显示)
  - [x] scheduler.py，pipeline.py
- [x] Scheduler.visualize_frames加了可视化选项，matplotlib在需要可视化时才导入，真实场景无需导入。pipeline和协同测试都正常
  - [x] scheduler.py，pipeline.py
- [x] 修复了Scheduler.simulate_pipeline的bug，即Worker2已经完成了所有层时，Worker3就不需要传输和计算了。pipeline和协同测试正常
  - [x] scheduler.py
- [x] 重构了Master，Worker，Trainer的rpc部分，IFR移动到了core，使得所有请求都通过stub_factory中的`*Stub`来序列化，通过`*Servicer`来反序列化。协同测试正常
  - [x] ifr.py：因为IFR同时被Master和Worker引用，所以把IFR从worker移动到core
  - [x] master.py，master_servicer.py，msg.proto：check_result改成report_finish，完成时总是上报，需要检查结果时上传Tensor
  - [x] scheduler.py：IFR移动，相应修改
  - [x] stub_factory.py：把Master，Worker，Trainer的发出请求过程全部集中在这里
  - [x] worker.py，worker_servicer.py：IFR移出去了，序列化相关的全部移出去了
- [x] Master新增了对IFR的计时。协同测试正常
  - [x] master.py
- [x] Master新增了配置项`pd_num`和`itv_num`，把保存PendingIpt的deque改成了线程安全的Queue。协同测试正常
  - [x] master.py，config.yml
- [x] 把print替换成logger。pipeline和协同测试正常
  - [x] config.yml：配置文件改成调试状态
  - [x] logging.yml：添加了相应配置
  - [x] master.py，master_servicer.py，scheduler.py，trainer.py，trainer_servicer.py，worker.py，worker_servicer.py：print改logger，需要logger的静态函数添加默认处理，Trainer代码风格整理
  - [x] util.py：cached_func使用logger输出，添加了缓存路径的选项
- [x] Master添加了平均IFR时延的统计。协同测试正常
  - [x] master.py
- [x] Scheduler修复bug：只对有改进的方案进行可视化。pipeline测试正常
  - [x] scheduler.py，pipeline.py

## 2021.12.5

- [x] Scheduler新增了从负载均衡方案开始局部搜索的函数optimize_chain。协同测试可以优化得到结果(正确性不确定)，但因为IntegralExecutor不支持空任务，最后一个Worker会崩溃
  - [x] scheduler.py
- [x] IntegralExecutor增加了对空任务的支持，新增了相关的单元测试。单元测试正常，协同测试正常
  - [x] integral_executor.py，test_integral_executor.py，test_dif_executor.py
- [x] Scheduler的配置中添加了带宽。协同测试正常
  - [x] config.yml，master.py，scheduler.py
- [x] Scheduler单元测试小bug修复
  - [x] test_scheduler.py
- [x] Scheduler的耗时估计单独抽象出一个函数simulate_pipeline，修复了之前的bug。使用pipeline.py对其进行可视化，但还没对ax_pc进行可视化。pipeline可视化正常，协同测试正常
  - [x] scheduler.py，pipeline.py

## 2021.12.4

- [x] 确保Worker Id从0开始连续增加，计算能力的baseline用worker0，协同运行正常
  - [x] main.py：加载config时检查
  - [x] master.py：传参数改用list
  - [x] scheduler.py：dict改成list
  - [x] config.yml：注释修改

## 2021.12.3

- [x] 新增了Scheduler.estimate_latency函数用于估计耗时，协同测试正常，但还没测试这个函数的正确性
  - [x] scheduler.py

## 2021.12.2

- [x] 添加了视频数据：campus，parking

> 视频数据来源：[The VIRAT Video Dataset](https://viratdata.org/)的release2.0
>
> | 本仓库中的视频名 | VIRAT中的视频名                 |
> | ---------------- | ------------------------------- |
> | campus           | VIRAT_S_010204_01_000072_000225 |
> | parking          | VIRAT_S_050201_05_000890_000944 |
> | road             | VIRAT_S_050000_13_001722_001766 |

## 2021.12.1

- [x] Scheduler添加了split_chain函数，用来对链状CNN按照Worker的计算能力进行均匀切割，使得耗时尽可能接近；Master执行的IFR数加入了config；DAG网页生成在当前执行路径下；其他小修改。单元测试和本地测试正常
  - [x] scheduler.py，test_scheduler.py：添加了split_chain函数，修改了初始化，添加了单元测试
  - [x] config.yml，master.py：IFR加入配置文件
  - [x] raw_dnn.py：DAG网页生成路径改到了当前路径
- [x] 对Scheduler的split_chain进行了修改，单元测试和协同测试正常
  - [x] scheduler.py，test_scheduler.py
- [x] :four_leaf_clover: [实验] lfcnz_show的sz模式添加了显示数据量预测的error

## 2021.11.29

- [x] SizedNode中添加了nz_thres，Scheduler添加了lcnz2lsz函数，实验进行了相应修改。lfcnz_show和协同运行均正常
  - [x] scheduler.py：SizedNode中添加了nz_thres，Scheduler添加了lcnz2lsz函数
  - [x] master.py：把wk_costs传入Scheduler构造函数
  - [x] lfcnz_show.py：相应修改
  - [x] config.yml：方便起见，默认配置使用480x720分辨率

## 2021.11.24

- [x] :four_leaf_clover: [实验] lfcnz_show的sz模式展示各层原始和压缩后的数据量
  - [x] experiments/lfcnz_show.py

## 2021.11.23

- [x] :four_leaf_clover: [实验] lfcnz_show.py展示三个图：误差，真实值，预测值
  - [x] experiments/lfcnz_show.py
- [x] 修复了Scheduler在对DNN进行predict时的bug：没有收集到所有前驱的数据就进行了预测。lfcnz_show在GoogLeNet上运行正常
  - [x] scheduler.py：bug修复
  - [x] lfcnz_show.py：GoogLeNet的数据集暴露了这个问题

## 2021.11.22

- [x] 使用Trainer和WorkerProfiler对离线数据收集过程进行抽象，torch==1.4.0+cpu且torchvision==0.5+cpu下master输出误差在1e-7左右

  * 注：之前误差比这个大，是因为版本较高（torch==1.9），而且这个版本下测试时执行Module会直接崩溃，所以库版本应该以 torch==1.4.0+cpu，torchvision==0.5+cpu 为准

  - [x] .gitignore：屏蔽缓存文件格式
  - [x] config.yml：添加了Trainer和WorkerProfiler相关配置
  - [x] dif_executor.py：DifExecutor使用了泛型type hint
  - [x] executor.py：Node定义移动到这里，Executor使用了泛型type hint
  - [x] integral_executor.py：IntegralExecutor使用了泛型type hint
  - [x] main.py：添加了Trainer的启动命令
  - [x] master.py：启动时添加了获取输出数据大小、获取Worker计算能力、获取Predictor的过程
  - [x] master_servicer.py：Master传参改成了直接传入全局config
  - [x] msg.proto：添加了profile相关的rpc
  - [x] node.py：因为Node功能大大简化，所以删掉了
  - [x] raw_dnn.py：把RawLayer转Node的代码移动到了Node类中
  - [x] scheduler.py：添加了对输出数据大小的profile代码，因为这个只有Scheduler会用到
  - [x] stub_factory.py：添加了Trainer的client
  - [x] trainer.py：运行在高性能服务器上，收集稀疏数据，训练预测模型
  - [x] trainer_servicer.py：为Trainer提供rpc服务
  - [x] worker.py：Worker初始化直接读取全局配置，添加了获取每层耗时的rpc接口
  - [x] worker_profiler.py：对Worker各层执行耗时进行统计
  - [x] worker_servicer.py：添加了获取每层耗时的rpc接口
  
- [x] :four_leaf_clover: experiments添加了gn197.py，用于查看GoogLeNet第197层预测不准的原因​ 

  - [x] experiments/gn197.py

- [x] [环境说明] torch1.4没有count_nonzero这个API，所以应该使用 torch==1.7.0+cpu且torchvision==0.8.1+cpu，该配置下master输出误差在1e-6到3e-6左右
* 注：这个版本是有count_nonzero的最低版本

- [x] 把代码分类整理到了多个目录中，运行正常
- [x] 把rpc相关代码整理到了目录rpc下面，测试运行正常
- [x] 把profile缓存抽象成通用的util，用于对Worker和Trainer的profile结果进行缓存
  - [x] trainer.py：cached_func移到core.util中
  - [x] util.py：通用的结果缓存函数
  - [x] worker.py：把WorkerProfiler的代码放到了这里
  - [x] worker_profiler.py：删除了
  - [x] worker_servicer.py：添加日志输出
- [x] 使用util.cached_func对Master的profile结果进行缓存，测试正常
  - [x] master.py：添加了缓存，去掉了存在误差的TODO
  - [x] scheduler.py：SizedNode.raw2dag函数重命名，以免与Node同名函数冲突
- [x] 添加了使用Predictor进行预测的代码，测试正常
  - [x] master.py：修复了没有反序列化的bug
  - [x] master_servicer.py：添加了调试输出
  - [x] scheduler.py：添加了预测代码
  - [x] worker_servicer.py：调试输出修改
- [x] :four_leaf_clover: [实验] lfcnz_show.py添加了对各层输出数据精度整体的可视化
  - [x] experiments/lfcnz_show.py
  - [x] scheduler.py：type hint的相关修改

## 2021.11.17

- [x] 使用Predictor对稀疏率预测进行抽象，去掉了LRD的部分，使用experiments中的lfcnz_show测试正常
  - [x] predictor.py：常用稀疏预测模型
  - [x] dnn_models的googlenet.py, resnet.py：做了相应修改
  - [x] experiments/lfcnz_show.py：改用了Predictor进行预测
  - [x] dnn_config.py：去掉了LRD部分，添加了Predictor
  - [x] raw_dnn.py：去掉了LRD部分
  - [x] lrd.py：删掉了
  - [x] node.py：去掉了LRD部分

## 2021.11.15

- [x] 把IntegralExecutor传入的参数从dnn_loader改成了RawDNN，Executor不再负责正确性检查，单元测试正常

## 2021.11.13

- [x] :four_leaf_clover: 添加了experiments目录，添加了LFPNZ格式数据的生成和可视化脚本
  - [x] experiments下的lfpnz_gen.py，lfpnz_show.py：生成和可视化脚本

- [x] :four_leaf_clover: ​LFPNZ数据可视化脚本添加了对DAG结构DNN的支持，GoogLeNet和ResNet50可视化正常
  - [x] experiments/lfpnz_show.py
- [x] :four_leaf_clover: experiments/lfpnz_show.py中添加了每个通道一个感知机的方案MLPs，发现：当输出通道少的时候，这个方案表现不如所有通道共用一个感知机的方法MLP，但是当输出通道多的时候（ResNet50后面的一些层），MLP效果极差，但是MLPs效果极好
- [x] :four_leaf_clover: 把关于“平面”称呼改成“通道”，本开发笔记所有experiments的提交说明都以:four_leaf_clover:开头​
- [x] msg.proto注释中的“平面”改成“通道”

## 2021.11.11

- [x] 修复了WorkerServicer的一个小bug，Master和3个Worker协同测试误差和之前一样
  - [x] scheduler.py：AlexNet在3个Worker上的执行计划
  - [x] worker_servicer.py：bug修复
  - [x] config.yml：添加了一个Worker
- [x] 把Servicer的RPC client封装成了StubFactory，Master和3个Worker协同测试误差和之前一样
  - [x] stub_factory.py：封装Client的获取过程
  - [x] master.py，master_servicer.py，worker.py，worker_servicer.py：调RPC方式相应修改

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

- [x] 把DNN配置参数封装成了DNNConfig，raw_layers封装成了RawDNN，单元测试全部通过，Master和1个Worker测试误差仍然在3e-6到5e-6左右
  - [x] dnn_config.py：DNNConfig所需的各种接口类
  - [x] raw_dnn.py：根据DNNConfig可以获取的原始类，可以执行，对外隐藏了不必要的API
  - [x] dnn_models/：chain.py，googlenet.py，resnet.py做了相应修改
  - [x] node.py：相应修改
  - [x] integral_executor.py：相应修改
  - [x] lrd.py：相应修改
  - [x] master.py：相应修改

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
  - [x] dnn_dag.py：`__make_dag`生成RawLayer后，把所有的ReLU.inplace置为False
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