# 2024 DAC SDC 智能驾驶挑战赛 - Team GO3 解决方案

## 📄 许可证
本项目采用限制性许可证，仅允许用于技术展示和学习目的。**禁止任何形式的商业使用、修改或二次分发**  
完整条款请查看 [LICENSE](LICENSE) 文件

## 🗂 项目结构
```
.
├── deploy/                # 部署相关
│   ├── deploy.ipynb       # Jupyter Notebook部署脚本
│   └── model.trt          # 带预处理和后处理的TensorRT模型
├── preprocess/            # 数据预处理
│   └── sdcexport.py       # 添加神经网络预处理层
├── scripts/               # 工具脚本
│   └── dataset_transform_script.py  # 数据集格式转换
└── yolov5/                # 改进版YOLOv5
    ├── network.yaml       # 网络结构定义
    ├── hyp.yaml          # 超参数配置
    ├── common.py         # 修改支持RepVGG
    └── yolo.py           # 修改支持RepVGG
```

## 🚀 快速开始

### 环境要求
- JetPack 4.6.1
- CUDA 10.2
- TensorRT 8.2.1
- Python 3.6.9

### 主要改进
1. **RepVGG支持**
   - 替换YOLOv5原始卷积模块为RepVGGBlock
   ```python
   # 详见 yolov5/common.py 中的 RepVGGBlock 实现
   ```

2. **矩形训练优化**
   - 修改letterbox方法支持288x512分辨率
   - 调整Mosaic数据增强以适应矩形输入

### 训练流程
1. 数据集准备
```bash
python scripts/dataset_transform_script.py --source [原始数据集路径] --dest [输出路径]
```

2. 模型训练
```bash
python train.py --img 512 --batch 16 --cfg yolov5/network.yaml --data [数据集yaml] --weights yolov5s.pt
```

3. 模型导出
```bash
python preprocess/sdcexport.py --weights [训练权重].pt --include onnx
```

### Jetson Nano部署
1. 转换TensorRT引擎
```python
from deploy.deploy import BaseEngine
engine = BaseEngine(onnx_path="yolov5s.onnx")
engine.serialize("model.trt") 
```

2. 运行推理
```bash
jupyter notebook deploy/deploy.ipynb
```

## ⚠️ 重要说明
1. 必须使用Jetson Nano进行最终部署，PC端仅用于训练
2. 模型输入分辨率固定为288x512
3. 依赖TensorRT-For-YOLO-Series后处理库 ([GitHub链接](https://github.com/Linaom1214/TensorRT-For-YOLO-Series))

## 📚 参考实现
- YOLOv5 v6.0 基础框架
- RepVGG 重参数化方案
- TensorRT-For-YOLO-Series 后处理优化