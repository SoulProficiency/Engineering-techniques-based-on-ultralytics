import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

# ==================== 配置参数 ====================
ONNX_MODEL = r'./yolo26n_640_6head.onnx'  # 您的分割ONNX模型路径
RKNN_MODEL = r'./yolo26n_640_6head.rknn'  # 输出的RKNN模型路径
DATASET = r'./dataset.txt'  # 量化数据集

QUANTIZE_ON = True


def export_rknn_inference():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # 注意：Letterbox 是自定义预处理，所以 config 中的 mean/std 仍然保留，
    # 但因为我们手动处理了 resize 和归一化，这里的 mean/std 会在 inference 时
    # 自动应用到输入 img 上。
    # 如果我们在传入 inference 前已经手动归一化了，需要确保这里的配置匹配。
    # 通常做法：输入 0-255 图像，让 RKNN 做归一化。
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588',
                quantized_dtype='w8a8',  # 或 'float16' 等
                quantized_algorithm='normal',
                quantized_method='channel',
                # 混合量化配置：指定哪些层不量化
                )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=["reg1", "cls1", "reg2", "cls2", "reg3", "cls3"])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    # print('--> Running model')
    # # 注意：如果 config 中设置了 std=255，这里的 img 应该是 0-255 的值
    # outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')


if __name__ == "__main__":
    export_rknn_inference()
