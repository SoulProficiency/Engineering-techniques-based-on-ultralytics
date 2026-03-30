# ReLU 与 SiLU 激活函数对比分析
## 为了进一步加速，我们尝试用ReLU替代SiLU

### 操作流程
#### 1.修改ultralytics/nn/modules/conv.py Line49
```
    # 这个方式适合直接导出，我们需要在训练的适合也修改为relu激活！！！
    default_act = nn.ReLU()  # default activation
```
#### 2.其余流程保持一致
```
整个流程目前已经很通透了，我们只需要在自己的目标数据集上更换nn.ReLU激活，并重新走一遍转换rknn的流程即可
```
#### 3.结果展示
```
推理 : 0.010835488319396973s , 后处理 : 0.001534026861190796s
```
![项目截图](./result/result_2head_rknn.jpg "系统界面截图")