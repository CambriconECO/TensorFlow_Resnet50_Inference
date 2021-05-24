# TensorFlow_Resnet50_Inference

#### 介绍

此demo演示了Cambricon TensorFlow框架下的resnet50模型在MLU270上的移植。

本示例基于Neuware 1.6.1+ Python3.5 版本测试通过。

相关教程链接：

#### 1.ckpt转pb

```
该步骤需要在Cambricon TensorFlow源码目录中进行，详细步骤参考教程：
如果没有源码，demo中已经提供了转换后的resnet50_v1.pb
```

#### 2.CPU推理

```
在tf_resnet50_v1目录下执行：
python tf_forward.py --input_pb resnet50_v1.pb --mode cpu
```

#### 3.模型量化

```
在tf_resnet50_v1目录下执行：
cp resnet50_v1_quant_param.txt_param resnet50_v1_quant_param.txt    
python tf_forward.py --input_pb resnet50_v1.pb --mode quant      
```

#### 4.MLU在线推理

```
在tf_resnet50_v1目录下执行：

# 在线逐层
python tf_forward.py --input_pb resnet50_v1.pb --mode online_layer

# 在线融合
python tf_forward.py --input_pb resnet50_v1.pb --mode online_fusion
```

#### 5.MLU离线推理

##### 生成离线模型

```
在tf_resnet50_v1目录下执行：
python tf_forward.py --input_pb resnet50_v1.pb --mode offline
```

##### 离线推理

```
在cnrt_resnet50_demo目录下执行：
# 编译
make
# 运行
./cnrt_resnet50_demo ../tf_resnet50_v1/resnet50_v1.cambricon subnet0 0 0 ../tf_resnet50_v1/fox.jpg 1
```

