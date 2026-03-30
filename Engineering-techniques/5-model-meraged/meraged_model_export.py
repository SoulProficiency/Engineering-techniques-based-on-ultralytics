import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
import onnxslim
class FeatureExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = {}
        for i, m in enumerate(self.layers):
            f = getattr(m, 'f', -1)
            if f == -1:
                x_in = x
            elif isinstance(f, list):
                x_in = [x if idx == -1 else outputs[idx] for idx in f]
            else:
                x_in = outputs[f]

            x = m(x_in)
            idx = m.i if hasattr(m, 'i') else i
            outputs[idx] = x
        return outputs

class MergedModel(nn.Module):
    def __init__(self, shared_layers, head1, head2):
        super().__init__()
        self.shared_layers = FeatureExtractor(shared_layers)
        self.head1 = head1
        self.head2 = head2

    def forward(self, x):
        all_outputs = self.shared_layers(x)
        for k,v in enumerate(all_outputs):
            print(k,v,all_outputs[v].shape)
        idx1 = self.head1.f
        idx2 = self.head2.f
        print("idx1 : ",idx1)
        print("idx2 : ",idx2)
        features1 = [all_outputs[i] for i in idx1]
        features2 = [all_outputs[i] for i in idx2]
        print(self.head1)
        print(self.head2)
        box1, cls1 = self.head1(features1)
        box2, cls2 = self.head2(features2)
        return box1, cls1, box2, cls2

# 加载模型
old_model = YOLO("../weights/yolo26n.pt")
new_model = YOLO("path to your model.pt trained by training_.py")
old_model.eval()
new_model.eval()

# 提取共享层和头
shared_layers = old_model.model.model[:-1]   # 注意：YOLO 的 model 属性下还有一层 model
head1 = old_model.model.model[-1]
head2 = new_model.model.model[-1]

merged_model = MergedModel(shared_layers, head1, head2)
merged_model.eval()

# 冻结共享层
for param in merged_model.shared_layers.parameters():
    param.requires_grad = False

# 导出 ONNX
width = 384
dummy_input = torch.randn(1, 3, width, 640)
export_name = f"merged_dual_head_{width}.onnx"
with torch.no_grad():
    torch.onnx.export(
        merged_model,
        dummy_input,
        export_name,
        input_names=['images'],
        output_names=['box1', 'cls1', 'box2', 'cls2'],
        opset_version=17,
        # dynamic_axes={'images': {0: 'batch'}, 'boxes1': {0: 'batch'}, 'scores1': {0: 'batch'},
        #               'boxes2': {0: 'batch'}, 'scores2': {0: 'batch'}}
    )

try:
    # 加载刚导出的模型
    onnx_model = onnx.load(export_name)
    # 执行简化
    slimmed_model = onnxslim.slim(onnx_model)
    # 保存简化后的模型 (可以覆盖原文件或另存为新文件)
    onnx.save(slimmed_model, export_name)
    print(f"ONNX model simplified and saved to {export_name}")
except Exception as e:
    print(f"Simplification failed: {e}. The original model is kept.")
print(f"{export_name} export success!")