import cv2
import numpy as np
import onnxruntime as ort
import time


class YOLO26DualHead_Flat:
    def __init__(self, model_path: str, conf_thres: float = 0.5, max_det: int = 100):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_thres = conf_thres
        self.max_det = max_det

        # 获取输入尺寸
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (H, W)

        # 获取输出信息 (预期有4个输出: box1, cls1, box2, cls2)
        outputs_info = self.session.get_outputs()
        if len(outputs_info) < 4:
            raise ValueError(f"预期模型有4个输出 (box1, cls1, box2, cls2), 但找到 {len(outputs_info)} 个输出。")

        # 解析输出形状
        # 假设导出顺序为: output0(box1), output1(cls1), output2(box2), output3(cls2)
        # 也可以通过名称查找，这里假设顺序固定
        self.box1_shape = outputs_info[0].shape  # (1, N, 4)
        self.cls1_shape = outputs_info[1].shape  # (1, N, nc1)
        self.box2_shape = outputs_info[2].shape  # (1, N, 4)
        self.cls2_shape = outputs_info[3].shape  # (1, N, nc2)

        self.nc1 = self.cls1_shape[-1]  # Head1 类别数 (e.g., 80)
        self.nc2 = self.cls2_shape[-1]  # Head2 类别数 (e.g., 4)
        self.num_anchors = self.box1_shape[1]  # 锚点数 (e.g., 8400)

        print(f"Model Input: {self.input_shape}")
        print(f"Head 1 -> Anchors: {self.num_anchors}, Classes: {self.nc1}")
        print(f"Head 2 -> Anchors: {self.num_anchors}, Classes: {self.nc2}")

        # 预生成解码网格 (两个Head共享特征层，grid只需生成一次)
        self.grids, self.strides = self._generate_grids()

    def _generate_grids(self):
        """预计算 anchor 的 grid 坐标和 stride"""
        grid_list = []
        stride_list = []
        strides = [8, 16, 32]  # YOLO 默认 strides

        for stride in strides:
            h = self.input_shape[0] // stride
            w = self.input_shape[1] // stride
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            # 坐标 + 0.5 (中心点)
            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2) + 0.5
            grid_list.append(grid)
            stride_list.append(np.full((h * w, 1), stride, dtype=np.float32))

        return np.concatenate(grid_list, axis=0), np.concatenate(stride_list, axis=0)

    def letterbox(self, img: np.ndarray, new_shape=(640, 640)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, dw, dh

    def _postprocess_single_head(self, boxes_raw, scores_raw, ratio, dw, dh, nc):
        """
        单个 Head 的后处理逻辑
        Args:
            boxes_raw: (1, N, 4) - 假设是 ltrb 格式相对值
            scores_raw: (1, N, nc)
        """
        # 移除 batch 维度 -> (N, 4)
        boxes = boxes_raw[0].copy()  # copy 避免修改原数据
        scores = scores_raw[0]

        # === 1. 坐标解码 ===
        # 使用预计算的 grids 和 strides 进行广播计算
        # boxes 输出格式假设为 ltrb (距离中心点的上下左右距离)
        # x1 = (grid_cx - l) * stride
        # x2 = (grid_cx + r) * stride
        boxes[:, [0, 2]] = (self.grids[:, [0]] + np.array([-1, 1]) * boxes[:, [0, 2]]) * self.strides
        boxes[:, [1, 3]] = (self.grids[:, [1]] + np.array([-1, 1]) * boxes[:, [1, 3]]) * self.strides

        # === 2. 置信度过滤 ===
        # scores: (N, nc)
        scores_max = scores.max(axis=1)
        mask = scores_max > self.conf_thres

        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        confidences = scores_max[mask]
        class_ids = scores.argmax(axis=1)

        # === 3. TopK (One2One) ===
        if boxes.shape[0] > self.max_det:
            topk_indices = np.argsort(confidences)[::-1][:self.max_det]
            boxes = boxes[topk_indices]
            confidences = confidences[topk_indices]
            class_ids = class_ids[topk_indices]

        # === 4. 还原坐标到原图 ===
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio

        # 组合结果: [x1, y1, x2, y2, conf, cls_id]
        return np.concatenate([boxes, confidences[:, None], class_ids[:, None].astype(np.float32)], axis=1)

    def inference(self, img_tensor, ratio, dw, dh):
        t1 = time.time()
        # 执行 ONNX 推理
        outputs = self.session.run(None, {'images': img_tensor})
        t2 = time.time()

        # outputs 顺序: [box1, cls1, box2, cls2]
        # 注意：如果导出时 output_names 顺序不对，这里可能需要调整索引
        out_box1 = outputs[0]
        out_cls1 = outputs[1]
        out_box2 = outputs[2]
        out_cls2 = outputs[3]

        # 分别处理两个 Head
        results_head1 = self._postprocess_single_head(out_box1, out_cls1, ratio, dw, dh, self.nc1)
        results_head2 = self._postprocess_single_head(out_box2, out_cls2, ratio, dw, dh, self.nc2)

        t3 = time.time()
        print(f"Infer: {t2 - t1:.4f}s, Post: {t3 - t2:.4f}s")

        # 返回字典，方便区分
        return {
            'head1': results_head1,  # shape: (N, 6)
            'head2': results_head2  # shape: (M, 6)
        }


if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO26DualHead_Flat(r"../weights/merged_dual_head_384.onnx")  # 请替换为你的模型路径

    # 2. 准备输入图像
    img = cv2.imread("../test_img/test2.jpg")
    if img is None:
        print("Error: Image not found.")
        exit()

    img_input, ratio, dw, dh = model.letterbox(img, model.input_shape)
    img_input = img_input[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # 3. 推理
    results = model.inference(img_input, ratio, dw, dh)

    # 4. 绘制结果
    # 定义两个 Head 的类别名称（可选）
    names_head1 = [f"COCO_{i}" for i in range(model.nc1)]
    names_head2 = [f"NEW_{i}" for i in range(model.nc2)]

    # 绘制 Head 1 结果 (例如用蓝色框)
    for *xyxy, conf, cls in results['head1']:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(cls)
        print(x1,y1,x2,y2)
        label = f"H1-{names_head1[cls_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 绘制 Head 2 结果 (例如用绿色框)
    for *xyxy, conf, cls in results['head2']:
        x1, y1, x2, y2 = map(int, xyxy)
        print(x1, y1, x2, y2)
        cls_id = int(cls)
        label = f"H2-{names_head2[cls_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色
        cv2.putText(img, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 文字画在框下面一点以免重叠

    cv2.imwrite("../infer_results/result_dual_head.jpg", img)
    print("Saved to result_dual_head.jpg")
