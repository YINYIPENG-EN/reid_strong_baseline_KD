本项目reid采用**表征学习和度量学习**结合的方式进行训练

是在**reid-strong-baseline**基础上实现

度量学习采用三元组损失函数

数据集：mark1501(将数据集mark1501放在data文件夹下)

baseline网络：支持Resnet系列，例如resnet18、resnet34、rensnet_ibn等

# Reid训练

```shell
python tools/train.py --model_name resnet50_ibn_a --model_path weights/ReID_resnet50_ibn_a.pth --IMS_PER_BATCH 8 --TEST_IMS_PER_BATCH 4 --MAX_EPOCHS 120
```

**model_name:**可支持的baseline网络

​						 支持：resnet18,resnet34,resnet50,resnet101,resnet50_ibn_a

接着会出现下面的内容：

```shell
=> Market1501 loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
  
2023-05-15 14:30:55.603 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[227/1484] Loss: 6.767, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:55.774 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[228/1484] Loss: 6.761, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:55.946 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[229/1484] Loss: 6.757, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:56.134 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[230/1484] Loss: 6.760, Acc: 0.000, Base Lr: 3.82e-05
2023-05-15 14:30:56.305 | INFO     | engine.trainer:log_training_loss:119 - Epoch[1] Iteration[231/1484] Loss: 6.764, Acc: 0.000, Base Lr: 3.82e-05

```

每个epoch训练完成后会测试一次mAP：

我这里第一个epoch的mAP达到75.1%，Rank-1:91.7%, Rank-5:97.2%, Rank-10:98.2%。

测试完成后会在log文件下保存一个pth权重，名称为mAPxx.pth，也是用该权重进行测试。

```shell
2023-05-15 14:35:59.753 | INFO     | engine.trainer:print_times:128 - Epoch 1 done. Time per batch: 261.820[s] Speed: 45.4[samples/s]
2023-05-15 14:35:59.755 | INFO     | engine.trainer:print_times:129 - ----------
The test feature is normalized
2023-05-15 14:39:51.025 | INFO     | engine.trainer:log_validation_results:137 - Validation Results - Epoch: 1
2023-05-15 14:39:51.048 | INFO     | engine.trainer:log_validation_results:140 - mAP:75.1%
2023-05-15 14:39:51.051 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-1  :91.7%
2023-05-15 14:39:51.051 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-5  :97.2%
2023-05-15 14:39:51.052 | INFO     | engine.trainer:log_validation_results:142 - CMC curve, Rank-10 :98.2%

```

# 知识蒸馏训练

支持网络为ResNet系列。

参数说明(基本参数和上面训练一样，只是多了kd)：

--model_name:模型名称，支持Resnet，resnet18_kd, resnet34_kd, resnet50_kd, resnet101_kd

--model_path:预权重路径

--kd:开启蒸馏模式

--feature_loss_coefficient:特征蒸馏的权重，默认0.03

这里用的蒸馏为在**线式蒸馏**(自蒸馏)，暂未更新离线式蒸馏。

```python
python tools/train.py --model_name [model name] --model_path [your model weight path] --IMS_PER_BATCH 8 --TEST_IMS_PER_BATCH 4 --kd --feature_loss_coefficient 0.03
```

```shell
=> Market1501 loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
resnet50_kd loading pretrained model weight...
label smooth on, numclasses: 751
ready kd train!

```

训练后会在logs文件下保存权重，命名格式为mAP_KD_xx.pth。下面是resnet50蒸馏前后**第一个Epoch**评价指标对比，还是有提升的【由于本人硬件环境受限，只是给大家把功能进行了实现】。

```
resnet50
Validation Results - Epoch: 1
2023-05-17 20:08:53.642 | INFO     | engine.trainer:log_validation_results:156 - mAP:39.2%
2023-05-17 20:08:53.642 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-1  :65.6%
2023-05-17 20:08:53.642 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-5  :80.3%
2023-05-17 20:08:53.642 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-10 :85.0%

resnet50_kd:[layer3作为教师网络]
2023-05-17 20:22:07.030 | INFO     | engine.trainer:log_validation_results:153 - Validation Results - Epoch: 1
2023-05-17 20:22:07.131 | INFO     | engine.trainer:log_validation_results:156 - mAP:47.9%
2023-05-17 20:22:07.131 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-1  :73.5%
2023-05-17 20:22:07.131 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-5  :85.7%
2023-05-17 20:22:07.139 | INFO     | engine.trainer:log_validation_results:158 - CMC curve, Rank-10 :88.9%
```



## 教师网络&学生网络选择

教师网络采用深层网络，浅层网络为学生网络。

具体的教师网络和学生网络的选择可以看engine/trainer.py第46行至58行。

此处默认resnet中的layer3为教师网络，layer1，layer2为学生网络。具体的效果可以根据自己实际任务去尝试。采用特征蒸馏，暂未更新逻辑蒸馏。

```python
        elif kd:
            score, feat, layer_out_feat = model(img)
            loss = loss_fn(score, feat, target)
            teacher_feature = layer_out_feat[1].detach()  # 取出教师层
            '''
            （rannge(idx,len(layer_out_feat))，中的idx可以决定哪个作为教师）
            idx=1表示layer4为教师网络,layer3,layer2,layer1为student
            idx=2表示layer3为教师网络,layer2,layer1为student
            idx=3表示layer2为教师网络，layer1为student
            '''
            for index in range(2, len(layer_out_feat)):  # layer4, layer3, layer2, layer1
                if index != 2:  # 排除自己
                    loss += torch.dist(layer_out_feat[index], teacher_feature) * feature_loss_coefficient
```



# 测试

```shell
python tools/test.py --TEST_IMS_PER_BATCH 4 --model_name [your model name] --model_path [your weight path]
```

可以进行mAP,Rank的测试

------

# Reid相关资料学习链接

数据集代码详解：https://blog.csdn.net/z240626191s/article/details/130371383?spm=1001.2014.3001.5501

Reid损失函数理论讲解：https://blog.csdn.net/z240626191s/article/details/130405664?spm=1001.2014.3001.5501

Reid度量学习Triplet loss代码讲解：https://blog.csdn.net/z240626191s/article/details/130490628?spm=1001.2014.3001.5501

**预权重链接：**

链接：https://pan.baidu.com/s/10dAj75wRiEZ7vuK8bU4GOg 
提取码：yypn

如果项目对你有用，麻烦点个Star

注：本项目暂时为免费开源，后期完善后会考虑适当收费【毕竟也是自己辛苦弄出来的】

# 后期计划更新

1.~~引入知识蒸馏训练~~(已于2023.05.26更新)

 2.加入YOLOX