import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from typing import Tuple, List


class YOLO26One2One_RKNNFlat:
    def __init__(self, model_path: str, input_shape: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.01, max_det: int = 300, num_classes: int = 80):
        self.rknn_lite = RKNNLite()
        self.conf_thres = conf_thres
        self.max_det = max_det
        self.input_shape = input_shape  # (H, W)
        self.nc = num_classes

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

        # 预生成解码网格和步长（基于三个尺度，总和8400）
        self.grids, self.strides = self._generate_grids()
        print(f"Model Inputs: {self.input_shape}, Anchors: {self.grids.shape[0]}, Classes: {self.nc}")

    def _generate_grids(self):
        """
        预计算每个 anchor 对应的网格坐标和步长。
        假设三个尺度：stride 8, 16, 32 对应特征图尺寸分别为 (48,80), (24,40), (12,20)
        输出：grids (8400,2), strides (8400,1)
        """
        grid_list = []
        stride_list = []

        # 根据输入尺寸计算各尺度步长和网格尺寸
        strides = [8, 16, 32]
        for stride in strides:
            h = self.input_shape[0] // stride
            w = self.input_shape[1] // stride

            # 生成网格坐标（中心点+0.5）
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((xv, yv), axis=-1).reshape(-1, 2).astype(np.float32) + 0.5

            grid_list.append(grid)
            stride_list.append(np.full((h * w, 1), stride, dtype=np.float32))

        # 拼接所有尺度的网格和步长
        grids = np.concatenate(grid_list, axis=0)  # (8400, 2)
        strides = np.concatenate(stride_list, axis=0)  # (8400, 1)
        return grids, strides

    def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """图像缩放与填充，保持宽高比，填充至 input_shape"""
        shape = img.shape[:2]  # [h, w]
        new_shape = self.input_shape[::-1]  # 输入为 (H,W)，letterbox需要 (W,H)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])  # 缩放比例（宽高分别计算，取较小值）

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, dw, dh

    def pre_process(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """预处理：letterbox -> BGR to RGB -> HWC to CHW -> 归一化 -> 添加 batch 维度"""
        img_input, ratio, dw, dh = self.letterbox(img)
        input_img = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        # 增加维度，img是对的
        input_img = np.expand_dims(input_img, 0)
        return input_img, ratio, dw, dh

    def postprocess(self, outputs: List[np.ndarray], ratio: float, dw: float, dh: float) -> np.ndarray:
        """
        后处理：接受两个输出头 [reg (1, 8400, 4), cls (1, 8400, nc)]
        返回检测结果 (N, 6) : x1, y1, x2, y2, conf, cls_id
        """
        # 移除 batch 维度
        boxes = outputs[0][0]  # (8400, 4)
        scores = outputs[1][0]  # (8400, nc)

        # 解码坐标（使用预生成的网格和步长）
        # boxes 为 ltrb 格式，解码公式: x = (grid_cx ± l/r) * stride
        boxes[:, [0, 2]] = (self.grids[:, [0]] + np.array([-1, 1]) * boxes[:, [0, 2]]) * self.strides
        boxes[:, [1, 3]] = (self.grids[:, [1]] + np.array([-1, 1]) * boxes[:, [1, 3]]) * self.strides

        # 置信度过滤
        scores_max = scores.max(axis=1)
        mask = scores_max > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        confidences = scores_max[mask]
        class_ids = scores.argmax(axis=1)

        # TopK 限制（One2One 模式）
        if boxes.shape[0] > self.max_det:
            topk_indices = np.argsort(confidences)[::-1][:self.max_det]
            boxes = boxes[topk_indices]
            confidences = confidences[topk_indices]
            class_ids = class_ids[topk_indices]

        # 坐标还原到原图
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= ratio

        # 组装结果
        detections = np.concatenate([
            boxes,
            confidences[:, np.newaxis],
            class_ids[:, np.newaxis].astype(np.float32)
        ], axis=1)
        return detections

    def inference(self, img_input: np.ndarray, ratio: float, dw: float, dh: float) -> np.ndarray:
        """执行推理并返回检测结果"""
        count_ = 1000
        infer_ = []
        postprocess_ = []
        while count_ > 0:
            t1 = time.time()
            outputs = self.rknn_lite.inference(inputs=[img_input])
            t2 = time.time()
            detections = self.postprocess(outputs, ratio, dw, dh)
            t3 = time.time()
            infer_.append(t2 - t1)
            postprocess_.append(t3 - t2)
            count_ -= 1
        # print(f"Infer: {t2 - t1:.4f}s, Post: {t3 - t2:.4f}s")
        return detections,infer_,postprocess_


if __name__ == "__main__":
    MODEL_PATH = "./yolo26n_640_2head.rknn"  # 你的2输出头RKNN模型路径
    IMAGE_PATH = "./test2.jpg"

    model = YOLO26One2One_RKNNFlat(MODEL_PATH, input_shape=(640, 640), conf_thres=0.5, max_det=300)
    img = cv2.imread(IMAGE_PATH)

    if img is not None:
        img_input, ratio, dw, dh = model.pre_process(img)

        results,infer_,postprocess_ = model.inference(img_input, ratio, dw, dh)
        print(f"推理 : {sum(infer_) / 1000}s , 后处理 : {sum(postprocess_) / 1000}s")

        ctime = (sum(postprocess_) / 1000) + (sum(infer_) / 1000)
        for *xyxy, conf, cls in results:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "yolo26n infer+postprocess FPS : {}".format(int(1/ctime)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(img, "single core NPU_CORE_0", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        cv2.imwrite("result_2head_rknn.jpg", img)
        print("Saved to result_2head_rknn.jpg")
