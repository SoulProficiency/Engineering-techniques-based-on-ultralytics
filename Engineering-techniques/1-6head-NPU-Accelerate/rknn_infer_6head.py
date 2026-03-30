import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List
from rknnlite.api import RKNNLite


class YOLO26One2One_Separate:
    def __init__(self, model_path: str, conf_thres: float = 0.01, max_det: int = 300):
        self.rknn_lite = RKNNLite()

        # 1. 加载 RKNN 模型
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(ret)

        # 2. 初始化运行时环境（使用 NPU 核心 0）
        print('--> Init runtime environment')
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        self.conf_thres = conf_thres
        self.max_det = max_det

        # 获取输入尺寸
        self.input_shape = (640, 640)

        self.nc = 80
        print(f"Detected Classes: {self.nc}")

        # 预生成网格和步长 (直接根据索引对应)
        self.grids, self.strides = self._make_grid_and_stride()

    def _make_grid_and_stride(self):
        """预计算网格坐标，直接对应输出索引 0, 2, 4"""
        grids = []
        strides = []
        outputs_info = [(80, 80),(40, 40),(20, 20) ]

        # 3个尺度，对应索引 0, 2, 4
        for i in range(3):
            # 回归头形状: (1, 4, H, W)
            out_meta = outputs_info[i]
            h, w = out_meta
            # print(h,w)
            stride = self.input_shape[0] / h
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((xv, yv), 2).reshape(1, h, w, 2)

            grids.append(grid)
            strides.append(stride)
            # print(grid.shape)
            # print(stride)
        return grids, strides

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[
        np.ndarray, float, float, float]:
        """图像缩放与填充"""
        shape = img.shape[:2]  # [h, w]
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

    def postprocess(self, outputs: List[np.ndarray], ratio: float, dw: float, dh: float) -> np.ndarray:
        """
        后处理：适配固定顺序 6 输出头 (reg, cls, reg, cls, reg, cls)
        """
        all_boxes = []
        all_scores = []

        # 遍历 3 个尺度
        for i in range(3):
            # 直接索引获取数据
            reg_feat = outputs[i * 2]  # (1, 4, H, W)
            # print(reg_feat.shape)
            cls_feat = outputs[i * 2 + 1]  # (1, nc, H, W)
            # print(cls_feat.shape)

            grid = self.grids[i]
            stride = self.strides[i]

            # 转置为 (1, H, W, C) 方便处理
            # 使用 reshape 代替 transpose 可以稍微快一点点，但 transpose 逻辑更清晰
            box_data = reg_feat.transpose(0, 2, 3, 1)  # (1, H, W, 4)
            score_data = cls_feat.transpose(0, 2, 3, 1)  # (1, H, W, nc)

            # === 解码坐标 (Grid Center + 0.5) ===
            grid_cx = grid[..., 0] + 0.5
            grid_cy = grid[..., 1] + 0.5

            # box_data 预测的是 ltrb
            # 直接使用切片操作，避免多次索引
            l = box_data[..., 0]
            t = box_data[..., 1]
            r = box_data[..., 2]
            b = box_data[..., 3]

            # 计算 x1, y1, x2, y2
            x1 = (grid_cx - l) * stride
            y1 = (grid_cy - t) * stride
            x2 = (grid_cx + r) * stride
            y2 = (grid_cy + b) * stride

            # 组装 boxes
            boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)

            # 组装 scores (模型输出已包含 sigmoid)
            scores = score_data.reshape(-1, self.nc)

            all_boxes.append(boxes)
            all_scores.append(scores)

        # === 合并与后处理 ===
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        scores_max = scores.max(axis=1)
        mask = scores_max > self.conf_thres

        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        confidences = scores_max[mask]
        class_ids = scores.argmax(axis=1)

        # TopK (One2One 模式)
        if boxes.shape[0] > self.max_det:
            topk_indices = np.argsort(confidences)[::-1][:self.max_det]
            boxes = boxes[topk_indices]
            confidences = confidences[topk_indices]
            class_ids = class_ids[topk_indices]

        # === 坐标还原回原图 ===
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio

        detections = np.concatenate([
            boxes,
            confidences[:, np.newaxis],
            class_ids[:, np.newaxis].astype(np.float32)
        ], axis=1)

        return detections

    def inference(self, img_input: np.ndarray, ratio, dw, dh) -> np.ndarray:
        count_ = 1000
        infer_ = []
        postprocess_ = []
        while count_>0:
            t1 = time.time()
            outputs = self.rknn_lite.inference(inputs=[img_input])
    #        np.savez_compressed("rknn_outputs.npz", *outputs)
            t2 = time.time()
            detections = self.postprocess(outputs, ratio, dw, dh)
            t3 = time.time()
            infer_.append(t2 - t1)
            postprocess_.append(t3 - t2)
            count_ -= 1
        # print(f"Infer: {t2 - t1:.4f}s, Post: {t3 - t2:.4f}s")
        return detections,infer_,postprocess_

    def pre_process(self, img):

        img_input, ratio, dw, dh = self.letterbox(img, (640,640))
        input_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        # 增加维度，img是对的
        input_img = np.expand_dims(input_img, 0)
        return input_img, ratio, dw, dh


if __name__ == "__main__":
    MODEL_PATH = "./yolo26n_640_6head.rknn"
    IMAGE_PATH = "./test2.jpg"

    model = YOLO26One2One_Separate(MODEL_PATH, conf_thres=0.5, max_det=300)
    img = cv2.imread(IMAGE_PATH)

    if img is not None:
        img_input, ratio, dw, dh = model.pre_process(img)
        results,infer_, postprocess_= model.inference(img_input, ratio, dw, dh)
        # print(f"Detected {len(results)} boxes.")
        print(f"推理 : {sum(infer_) / 1000}s , 后处理 : {sum(postprocess_) / 1000}s")

        for *xyxy, conf, cls in results:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite("result_6head.jpg", img)
        print("Saved to result_6head.jpg")
