# 2head模式⭐⭐⭐⭐⭐
```
这里直接给出当前6Head模式存在的问题：
1.6Head模式的测试结果表现并不稳定
2.6head后处理模式存在可以合并优化的余地（NPU算子合并数据进一步降低CPU解码的压力）
```
## 1.导出2Head模型
### 修改修改ultralytics/nn/head.py Line161
```
    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            # 6head
            # print("6head")
            # y = []
            # for i in range(self.nl):
            #     t1 = self.cv2[i](x[i])
            #     t2 = self.cv3[i](x[i]).sigmoid()
            #     y.append(t1)
            #     y.append(t2)
            # return y

            print("2head")
            bs = x[0].shape[0]  # batch size
            box_list = []
            cls_list = []
            for i in range(self.nl):
                # 1. 获取原始输出
                box_feat = self.one2one_cv2[i](x[i])
                cls_feat = self.one2one_cv3[i](x[i]).sigmoid()  # 确保分类输出经过 sigmoid
                # 2. DFL 解码 (如果 reg_max > 1)
                # 如果模型是标准 YOLO (reg_max=16)，需要将 64通道 解码为 4通道
                if self.reg_max > 1:
                    b, c, h, w = box_feat.shape
                    # DFL 期望输入: (bs, 4*reg_max, N)
                    # view: (1, 64, H*W)
                    box_feat = box_feat.view(b, c, -1)
                    # dfl: (1, 4, H*W)
                    box_feat = self.dfl(box_feat)
                    # 还原空间维度以便后续转置: (1, 4, H, W)
                    box_feat = box_feat.view(b, 4, h, w)
                # 3. 维度变换与展平
                # (1, C, H, W) -> (1, H, W, C) -> (1, H*W, C)
                # Box: (1, 4, H, W) -> (1, H, W, 4) -> (1, H*W, 4)
                box_feat = box_feat.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
                # Cls: (1, nc, H, W) -> (1, H, W, nc) -> (1, H*W, nc)
                cls_feat = cls_feat.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.nc)
                box_list.append(box_feat)
                cls_list.append(cls_feat)
            # 4. 拼接所有层级 (Stride 8, 16, 32)
            # boxes: (1, 8400, 4)
            # scores: (1, 8400, nc)
            boxes = torch.cat(box_list, dim=1)
            scores = torch.cat(cls_list, dim=1)
            # 返回一个包含两个张量的列表，对应 ONNX 的两个输出头
            return [boxes, scores]
```
### onnx导出 执行onnx_export_2head.py
### 测试onnx模型 执行onnx_infer_2head.py

## 2.RKNN模型导出
```
    这一部分可以参考onnx_export_6head.py
    不再进行过多的赘述
```

## 3.2head rknn 推理
```
推理 : 0.03384818077087402s , 后处理 : 0.003440439462661743s
推理 : 0.03405026078224182s , 后处理 : 0.0031837556362152098s
推理 : 0.03531105327606201s , 后处理 : 0.0035695545673370362s
推理 : 0.03440890383720398s , 后处理 : 0.0034328584671020506s
推理 : 0.03443070197105408s , 后处理 : 0.0034598214626312255s
推理 : 0.03388308787345886s , 后处理 : 0.0034408173561096193s
```
# 2head 与 6head 模型性能对比分析

为了更直观地分析，将时间单位统一换算为**毫秒**进行计算和对比。

## 1. 数据统计概览

| 模型 | 阶段 | 最小值 | 最大值 | 平均值 | 方差 (ms²) | 标准差 |
| :--- | :--- | :--- | :--- | :--- |:---------| :--- |
| **2head** | 推理 | 33.84 | 35.31 | **34.32** | 0.24     | 0.49 |
| **6head** | 推理 | 32.63 | **36.95** | **34.34** | **2.13** | **1.46** |
| **2head** | 后处理 | 3.18 | 3.56 | **3.42** | 0.015    | 0.12 |
| **6head** | 后处理 | 2.85 | **5.07** | **3.73** | **0.53** | **0.72** |
| **2head** | **总耗时** | **37.02** | **38.87** | **37.74** | **0.26** | **0.51** |
| **6head** | **总耗时** | **35.48** | **42.02** | **38.07** | **3.06** | **1.75** |

## 2. 详细分析

### （1）平均速度（均值）
* **推理阶段**：两者几乎打平。`2head` 平均 34.32ms，`6head` 平均 34.34ms，差距可以忽略不计。
* **后处理阶段**：`2head`（3.42ms）略快于 `6head`（3.73ms）。
* **整体来看**：`2head` 的平均总耗时（37.74ms）比 `6head`（38.07ms）稍微快一点点，但差距极小（约0.3ms）。

### （2）稳定性（方差与标准差 —— 核心差异点）
* **推理阶段**：`6head` 的方差（2.13）是 `2head`（0.24）的近 **9倍**。`6head` 出现了明显的波动（比如突然飙升至 36.95ms），而 `2head` 的数据非常密集，全都稳定在 33.8ms - 35.3ms 之间。
* **后处理阶段**：`6head` 的方差（0.53）是 `2head`（0.015）的 **35倍**。`6head` 的后处理极不稳定，最低 2.85ms，最高却达到了 5.07ms（存在明显毛刺）；而 `2head` 的后处理极其平稳。

### （3）尾部延迟
* 在实际部署中，最差情况往往决定体验。`6head` 的最大总耗时达到了 **42.02ms**，而 `2head` 的最大总耗时仅为 **38.87ms**。`6head` 的极端慢情况比 `2head` 慢了约 3ms。

## 3. 结论：哪种模型更好？

**结论：在仅考虑推理和后处理时间的情况下，`2head` 模型明显更好。**

**理由：**
1. **胜在稳定**：在平均速度几乎一样的情况下，`2head` 展现出了碾压级的稳定性（方差极小）。在工程落地中，延迟的抖动往往比平均慢一点点更致命，会导致画面卡顿、音画不同步或请求超时。
2. **无毛刺**：`6head` 模型在推理和后处理阶段都出现了明显的“突刺”（毛刺），说明其计算过程可能受到了其他进程干扰，或者自身结构在处理某些数据时存在不稳定分支。
3. **尾部延迟更低**：`2head` 的最坏情况（38.87ms）也要好于 `6head` 的最坏情况（42.02ms）。

> **注**：当然，如果 `6head` 模型在准确率、召回率等业务指标上远超 `2head`，那么这 3~4ms 的延迟差距是可以接受的。但如果纯粹从性能和工程部署友好度来看，`2head` 是更优的选择，但是本例当中，两者进度差异并不明显。
