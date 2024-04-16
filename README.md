# InternLM2-Tutorial-Assignment-Lecture5  
# 第5课 XTuner 微调 LLM：1.8B、多模态和 Agent   
2024.4.11  XTuner 贡献者 李剑锋 汪周谦 王群    

[XTuner]( https://github.com/InternLM/XTuner)   
[第6课 视频]( https://b23.tv/QUhT6ni)   
[第6课 文档](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)   
[第6课 文档](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/readme.md)   
[第6课 作业](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/homework.md)    
[OpenXLab 部署教程](https://github.com/InternLM/Tutorial/tree/camp2/tools/openxlab-deploy)    

## 第6课 笔记   

### 1. 原理
- 为什么要微调调大模型： 现有的模型是基座模型 Foundation Model

- Finetune二种范式：增量预训练微调 和 指令跟随微调

- 一条数据的一生：
  - 原始数据
  - 标准格式数据 system user assistant
  - 添加对话模板  system input output 对话模版是为了能够让LLM区分出system user和assistant，不同的模型会有不同的模版。
  - Tokenized数据
  - 添加Label

![](./XTuner1.png)   
![](./XTuner2.png)
![](./XTuner3.png)
![](./XTuner4.png)   

- 微调的方案：
  - LoRA: Low-Rank Adapation of Large Lanaguage Models  比喻超大玩具（基底模型）中的某个零件(LoRA)进行改动
  - QLoRA 是LoRA的一种改进，类似有一把生锈的螺丝刀，也能改造玩具

![](./XTuner5.png)
![](./XTuner6.png)

### 2. XTuner   

- XTuner是以配置文件的形式专封装了大部分微调场景，对于0基础的非专业人员也能微调模型；轻量化，对于7B参数量的LLM，最小显存为8GB，消费级显卡就可以微调模型。

- XTuner适配多种生态，多种硬件.

- 与LLaMa-Factory相比，训练速度更快，微调效果更好。

- XTuner快速上手
  - 创建Conda环境
  - 安装XTuner: `pip install xtuner`
  - 挑选配置模板: `xtuner list-cfg -p internlm_20b`
  - 一键训练: `xtuner train internlm_20b_qlora_oasst1_512_e3`
  - 对话：Float16 模型对话 4bit模型对话 加载Adapter模型对话 工具类对话（网络搜素 计算器 解方程函数）  

- XTuner 数据引擎
  - 数据集映射函数 **开发者可以专注于数据内容，不必花费精力处理复杂的数据格式！**
  - 多数据拼接
 
![](./XTuner7.png)
![](./XTuner8.png)
![](./XTuner9.png)
![](./XTuner10.png)
![](./XTuner11.png)
![](./XTuner12.png)
![](./XTuner13.png)

### 3. 8GB显存玩转LLM

- XTuner二种加速优化方案：Flash Attention 和 DeepSpeed ZeRO
- 优化后，可以明显减少显存的占用。

### 4. InternLM2 1.8B 模型

- 三个版本开源模型：
    - InternLM2-1.8B : 基础模型，为下游深度适应提供了良好的起点；
    - InternLM2-chat-1.8B-SFT : 在InternLM2-1.8B 上进行监督微调（SFT）后得到的对话模型；
    - **InternLM2-Chat-1.8B** :   通过在线RLHF 在InternLM2-chat-1.8B-SFT 之上进一步对齐，表现出更好的指令跟随、聊天体验和函数调用，模型大小为3.78G,

### 5. 多模态LLM     

- 给LLM装上电子眼，多模态是识图，不是生图，
- LLaVA方案 Image Projector
- 快速上手

![](./XTuner14.png)
![](./XTuner15.png)


## XTuner 微调个人小助手认知

如何利用 XTuner 完成个人小助手的微调！

### 1 开发机准备

使用 `Cuda11.7-conda` 镜像，然后在资源配置中，使用 `10% A100 * 1` 的选项，创建开发机器。   

### 2 快速上手    

#### 2.1 环境安装    

- 安装一个 XTuner：`studio-conda xtuner0.1.17`
- 激活环境: `conda activate xtuner0.1.17`
- 进入家目录: `cd ~`
- 创建版本文件夹并进入: `mkdir -p /root/xtuner0117 && cd /root/xtuner0117`
- 拉取 0.1.17 的版本源码:  `git clone -b v0.1.17  https://github.com/InternLM/xtuner`
- 进入源码目录: ` cd /root/xtuner0117/xtuner`
- 从源码安装 XTuner: `pip install -e '.[all]'`
  
![](./XTuner16.png)    
![](./XTuner17.png)  

#### 2.2 前期准备    

##### 2.2.1 数据集准备

首先我们先创建一个文件夹来存放我们这次训练所需要的所有文件。    
```
# 前半部分是创建一个文件夹，后半部分是进入该文件夹。
mkdir -p /root/ft && cd /root/ft

# 在ft这个文件夹里再创建一个存放数据的data文件夹
mkdir -p /root/ft/data && cd /root/ft/data
```

在 data 目录下新建一个 generate_data.py 文件，将以下代码复制进去，然后运行该脚本即可生成数据集。
假如想要加大剂量让他能够完完全全认识到你的身份，那我们可以吧 `n` 的值调大一点。
```
# 创建 `generate_data.py` 文件
touch /root/ft/data/generate_data.py
```

打开该 python 文件后将下面的内容复制进去: 
```
import json

# 设置用户的名字
name = '不要姜葱蒜大佬'
# 设置需要重复添加的数据次数
n =  10000
.....
```
将文件 name 后面的内容修改为你的名称。
```
# 将对应的name进行修改（在第4行的位置）
- name = '不要姜葱蒜大佬'
+ name = "德林大佬"
```

修改完成后运行 generate_data.py 文件即可。    
```
# 确保先进入该文件夹
cd /root/ft/data

# 运行代码
python /root/ft/data/generate_data.py
```

##### 2.2.2 模型准备   

准备好了数据集后，接下来我们就需要准备好我们的要用于微调的模型。
小模型 `InterLM2-Chat-1.8B` 来完成此次的微调.


## 第6课 作业   

