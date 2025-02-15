# 2024 DAC SDC 智能驾驶挑战赛 - Team GO3 解决方案

## 🎯 项目概述
本项目实现了一个面向嵌入式设备的轻量级目标检测与实例分割系统。基于 YOLOv5 框架，我们通过一系列创新性的优化手段，在保证检测精度的同时显著提升了模型在 Jetson Nano 上的推理速度。

### 🏆 主要成果
- **比赛成绩**：[2024 DAC-SDC GPU赛道](https://pku-sec-lab.github.io/dac-sdc-2024/results-gpu/) 第二名
  * 精度指标：Precision 0.627，Recall 0.561，F1-Score 0.592，mIoU 0.1834
  * 性能指标：43.66 FPS
  * 总分：16.7659
- 完整的 GPU 加速流水线：从预处理到后处理的全流程 GPU 优化
- 创新性的道路线分割策略：采用多边形近似方法

### 💡 核心技术亮点
1. **轻量化网络设计**
   - 采用 RepVGG 重参数化技术作为骨干网络
   - 训练时保持多分支结构，推理时转换为单路径结构
   - 通过消融实验验证了相比 MobileNetV3 和 Darknet53 的优势

2. **优化的输入处理**
   - 采用 512x288 的矩形输入尺寸
   - 创新的右下角填充策略，简化图像恢复操作
   - GPU 加速的预处理操作，包括数据转置和归一化

3. **高效训练策略**
   - 多样化数据增强：Mosaic、尺寸缩放、左右翻转、对比度增强
   - 类别权重手动增强
   - Adam 优化器 + 余弦退火学习率调度

4. **推理加速优化**
   - TensorRT FP16 量化
   - Efficient NMS 插件实现 GPU 上的非极大值抑制
   - 全流程 GPU 计算，最小化 CPU-GPU 数据传输

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
1. **RepVGG 重参数化技术**
   - 创新性地将 RepVGG 应用于目标检测任务
   - 训练阶段：保持多分支结构，提升特征提取能力
   - 推理阶段：转换为单一卷积操作，显著提升计算效率
   - 通过结构重参数化，在不损失精度的情况下实现加速
   ```python
   # 详见 yolov5/common.py 中的 RepVGGBlock 实现
   ```

2. **矩形训练与推理优化**
   - 输入尺寸优化：采用 288x512 分辨率
     * 通过消融实验确定最佳输入尺寸
     * 平衡检测精度和推理速度
   - 创新的填充策略
     * 右下角定点填充，简化还原计算
     * 保持宽高比例，提升检测准确性
   - Mosaic 数据增强改进
     * 适配矩形输入的 Mosaic 实现
     * 增强小目标检测能力

3. **全流程 GPU 加速**
   - 预处理优化
     * 使用 CUDA 核函数实现图像预处理
     * 减少 CPU-GPU 数据传输开销
   - 后处理优化
     * TensorRT Efficient NMS 插件
     * FP16 精度推理加速
   - 内存优化
     * 减少中间结果的显存占用
     * 优化数据流转路径

### 训练流程
1. **数据集准备**
   ```bash
   python scripts/dataset_transform_script.py --source [原始数据集路径] --dest [输出路径]
   ```
   - 支持多种数据格式转换
   - 自动生成训练所需的标注文件
   - 数据集划分与统计分析

2. **模型训练**
   ```bash
   python train.py --img 512 --batch 16 --cfg yolov5/network.yaml --data [数据集yaml] --weights yolov5s.pt
   ```
   - 训练参数说明：
     * `--img`: 输入图像尺寸 (288x512)
     * `--batch`: 批次大小，根据显存调整
     * `--cfg`: 网络配置文件
     * `--data`: 数据集配置文件
     * `--weights`: 预训练权重
   - 训练过程监控：
     * 使用 TensorBoard 跟踪训练指标
     * 自动保存最佳模型权重

3. **模型导出**
   ```bash
   python preprocess/sdcexport.py --weights [训练权重].pt --include onnx
   ```
   - 支持多种导出格式：
     * ONNX 格式用于 TensorRT 转换
     * TorchScript 格式用于 C++ 部署
   - 自动添加预处理层
   - 模型结构优化与简化

### Jetson Nano 部署
1. **环境配置**
   - JetPack 4.6.1 安装
   - Python 依赖安装
   ```bash
   pip install -r requirements.txt
   ```

2. **TensorRT 引擎转换**
   ```python
   from deploy.deploy import BaseEngine
   
   # 创建 TensorRT 引擎
   engine = BaseEngine(
       onnx_path="yolov5s.onnx",
       fp16_mode=True,  # 启用 FP16 精度
       max_batch_size=1  # 设置最大批次大小
   )
   
   # 序列化引擎保存
   engine.serialize("model.trt")
   ```
   - 支持 FP16 量化加速
   - 内置 Efficient NMS 插件
   - 自动优化推理计算图

3. **模型推理**
   ```bash
   jupyter notebook deploy/deploy.ipynb
   ```
   - 提供完整的推理示例
   - 包含性能测试与评估
   - 可视化检测结果

## ⚠️ 重要说明
1. **部署环境要求**
   - 必须使用 Jetson Nano 进行最终部署
   - PC 端仅用于训练与模型转换
   - 确保 JetPack 和 CUDA 版本匹配

2. **模型约束**
   - 输入分辨率固定为 288x512
   - 仅支持单批次推理
   - 使用 FP16 精度以优化性能

3. **依赖说明**
   - 基于 TensorRT-For-YOLO-Series 后处理库
   - 需要 CUDA 支持的 OpenCV
   - 完整依赖列表见 requirements.txt

## 📚 参考实现
- YOLOv5 v6.0：目标检测基础框架
- RepVGG：高效神经网络结构设计
- TensorRT-For-YOLO-Series：[后处理加速库](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)
- NVIDIA TensorRT：深度学习推理优化