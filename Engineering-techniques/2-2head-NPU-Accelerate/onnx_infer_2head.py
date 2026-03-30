import cv2
import numpy as np
import onnxruntime as ort
import time


class YOLO26One2One_Flat:
    def __init__(self, model_path: str, conf_thres: float = 0.5, max_det: int = 100):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_thres = conf_thres
        self.max_det = max_det

        # 获取输入尺寸
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (H, W)

        # 获取输出信息
        # 输出1: (1, N, 4) -> reg
        # 输出2: (1, N, nc) -> cls
        out_reg_shape = self.session.get_outputs()[0].shape
        out_cls_shape = self.session.get_outputs()[1].shape

        self.nc = out_cls_shape[-1]  # 类别数
        self.num_anchors = out_reg_shape[1]  # 8400

        print(f"Model Inputs: {self.input_shape}, Anchors: {self.num_anchors}, Classes: {self.nc}")

        # 预生成解码网格
        # 因为模型内部只做了 flatten，没有做坐标解码，我们需要在推理端做
        # 但我们不需要在每一帧生成网格，只需在初始化时生成一次
        self.grids, self.strides = self._generate_grids()

    def _generate_grids(self):
        """预计算每个 anchor 对应的 grid 坐标和 stride"""
        # 这里的逻辑是反向推导每个 anchor 属于哪个尺度和网格
        # 假设导出顺序是 stride 8, 16, 32 (大特征图到小特征图)
        # 对应尺寸: input/8, input/16, input/32

        grid_list = []
        stride_list = []

        strides = [8, 16, 32]
        for stride in strides:
            h = self.input_shape[0] // stride
            w = self.input_shape[1] // stride

            # 生成网格
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            # 坐标 + 0.5 (中心点)
            # 展平为 (N, 2)
            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2) + 0.5

            grid_list.append(grid)
            # 每个 anchor 对应的 stride，方便后续计算
            stride_list.append(np.full((h * w, 1), stride, dtype=np.float32))

        # 拼接为 (8400, 2) 和 (8400, 1)
        return np.concatenate(grid_list, axis=0), np.concatenate(stride_list, axis=0)

    def letterbox(self, img: np.ndarray, new_shape=(640, 640)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2;
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, dw, dh

    def postprocess(self, outputs, ratio, dw, dh):
        # outputs: [reg(1, 8400, 4), cls(1, 8400, nc)]
        # 移除 batch 维度 -> (8400, 4)
        boxes = outputs[0][0]
        scores = outputs[1][0]

        # === 1. 坐标解码 ===
        # boxes 输出是 ltrb
        # x1 = (grid_cx - l) * stride
        # 这里的 grid 和 stride 是在 __init__ 中预计算好的 (8400, 2) 和 (8400, 1)

        # 利用广播机制一次性解码所有 8400 个框
        # grid: (8400, 2), boxes: (8400, 4), strides: (8400, 1)
        boxes[:, [0, 2]] = (self.grids[:, [0]] + np.array([-1, 1]) * boxes[:, [0, 2]]) * self.strides
        boxes[:, [1, 3]] = (self.grids[:, [1]] + np.array([-1, 1]) * boxes[:, [1, 3]]) * self.strides

        # === 2. 置信度过滤 ===
        scores_max = scores.max(axis=1)
        mask = scores_max > self.conf_thres

        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        confidences = scores_max[mask]
        class_ids = scores.argmax(axis=1)

        # === 3. TopK (One2One) ===
        # if boxes.shape[0] > self.max_det:
        #     topk_indices = np.argsort(confidences)[::-1][:self.max_det]
        #     boxes = boxes[topk_indices]
        #     confidences = confidences[topk_indices]
        #     class_ids = class_ids[topk_indices]

        # === 4. 还原坐标 ===
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio
        return np.concatenate([boxes, confidences[:, None], class_ids[:, None].astype(np.float32)], axis=1)

    def inference(self, img_tensor, ratio, dw, dh):
        count_ = 1000
        infer_ = []
        postprocess_ = []
        while count_>0:
            t1 = time.time()
            outputs = self.session.run(None, {'images': img_tensor})
            t2 = time.time()
            detections = self.postprocess(outputs, ratio, dw, dh)
            t3 = time.time()
            infer_.append(t2-t1)
            postprocess_.append(t3-t2)
            count_-=1
        return detections,infer_,postprocess_


# 使用示例
if __name__ == "__main__":
    model = YOLO26One2One_Flat("../weights/yolo26n_640_2head.onnx")
    img = cv2.imread("../test_img/test2.jpg")
    img_input, ratio, dw, dh = model.letterbox(img, model.input_shape)
    img_input = img_input[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # while True:
    results,infer_,postprocess_ = model.inference(img_input, ratio, dw, dh)
    print(f"推理 : {sum(infer_)/1000}s , 后处理 : {sum(postprocess_)/1000}s")
    # ... 绘制代码 ...
    print(f"obj : {len(results)}")
    for *xyxy, conf, cls in results:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{int(cls)}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("../test_img/result_2head.jpg", img)
    print("Saved to result_2head.jpg")