# 2024 DAC SDC 智能驾驶挑战赛 - Team GO3 解决方案

## 🎯 项目概述
本项目实现了一个面向嵌入式设备的轻量级目标检测与实例分割系统，专注于自动驾驶场景下的道路线检测任务。基于 YOLOv5 框架，我们通过一系列创新性的优化手段，在保证检测精度的同时显著提升了模型在 Jetson Nano 上的推理速度。

### 🏆 主要成果
- **比赛成绩**：[2024 DAC-SDC GPU赛道](https://pku-sec-lab.github.io/dac-sdc-2024/results-gpu/) 第二名
  * 精度指标：Precision 0.627，Recall 0.561，F1-Score 0.592，mIoU 0.1834
  * 性能指标：43.66 FPS（Jetson Nano 实时推理）
  * 总分：16.7659
- **技术创新**：
  * 全流程 GPU 加速：从预处理到后处理的端到端优化
  * 创新的分割策略：基于多边形近似的道路线分割
  * 轻量化网络设计：采用 RepVGG 结构重参数化

### 💡 核心技术亮点
1. **轻量化网络设计**
   - 创新性地将 RepVGG 应用于目标检测任务
   - 训练阶段：多分支结构提升特征提取能力
   - 推理阶段：单路径结构实现高效计算
   - 通过结构重参数化技术平衡精度与速度

2. **创新的道路线分割策略**
   - 多边形近似方法
     * 高精度与高性能的平衡
     * 显著降低计算复杂度
   - 优化的类别划分
     * 实线/虚线分别划分左右类别
     * 提升分割精度和召回率
   - 端到端分割流程
     * 无需复杂后处理
     * 直接输出分割结果

3. **优化的输入处理**
   - 输入尺寸优化
     * 采用 288x512 矩形输入分辨率
     * 通过消融实验确定最佳配置
   - 创新的填充策略
     * 右下角定点填充，简化还原计算
     * 保持宽高比例，提升检测准确性
   - GPU 加速预处理
     * CUDA 实现的数据转置和归一化
     * 最小化 CPU-GPU 数据传输

4. **高效训练策略**
   - 数据增强优化
     * 改进的 Mosaic 增强适配矩形输入
     * 多样化增强：尺寸缩放、翻转、对比度
   - 训练过程优化
     * 类别权重动态调整
     * Adam 优化器 + 余弦退火调度
     * 自动保存最佳模型检查点

5. **推理加速优化**
   - TensorRT 优化
     * FP16 精度量化加速
     * Efficient NMS 插件集成
     * 计算图自动优化
   - 内存优化
     * 减少中间结果显存占用
     * 优化数据流转路径
   - 全流程 GPU 计算
     * 最小化 CPU-GPU 数据传输
     * 流水线并行优化

## 📄 许可证
本项目采用限制性许可证，仅允许用于技术展示和学习目的。**禁止任何形式的商业使用、修改或二次分发**  
完整条款请查看 [LICENSE](LICENSE) 文件

## 🗂 项目结构
```
.
├── deploy/                # 部署相关
│   ├── deploy.ipynb       # Jupyter Notebook部署脚本
│   └── model.trt          # TensorRT优化模型
├── preprocess/            # 数据预处理
│   └── sdcexport.py       # 预处理层集成工具
├── scripts/               # 工具脚本
│   └── dataset_transform_script.py  # 数据集处理工具
└── yolov5/                # 改进版YOLOv5
    ├── network.yaml       # 网络结构配置
    ├── hyp.yaml          # 训练超参数
    ├── common.py         # RepVGG实现
    └── yolo.py           # 检测器实现
```

## 🚀 快速开始

### 环境配置
1. **硬件要求**
   - 训练：NVIDIA GPU（显存 ≥ 8GB）
   - 部署：Jetson Nano

2. **软件环境**
   - JetPack 4.6.1
   - CUDA 10.2
   - TensorRT 8.2.1
   - Python 3.6.9

3. **依赖安装**
   ```bash
   pip install -r requirements.txt
   ```

### 训练流程
1. **数据准备**
   ```bash
   python scripts/dataset_transform_script.py --source [原始数据集路径] --dest [输出路径]
   ```
   - 支持多种标注格式转换
   - 自动生成训练配置
   - 数据集统计分析

2. **模型训练**
   ```bash
   python train.py --img 512 --batch 16 --cfg yolov5/network.yaml --data [数据集yaml] --weights yolov5s.pt
   ```
   - 关键参数说明
     * `--img`: 输入尺寸 (288x512)
     * `--batch`: 批次大小
     * `--cfg`: 网络配置
     * `--data`: 数据配置
     * `--weights`: 预训练权重
   - 训练监控
     * TensorBoard 可视化
     * 自动保存最佳权重

3. **模型导出**
   ```bash
   python preprocess/sdcexport.py --weights [训练权重].pt --include onnx
   ```
   - 支持多种导出格式
   - 自动集成预处理
   - 模型结构优化

### 部署流程
1. **TensorRT 转换**
   ```python
   from deploy.deploy import BaseEngine
   
   engine = BaseEngine(
       onnx_path="yolov5s.onnx",
       fp16_mode=True,  # 启用FP16
       max_batch_size=1
   )
   engine.serialize("model.trt")
   ```
   - FP16 量化加速
   - NMS 插件优化
   - 计算图优化

2. **模型推理**
   ```bash
   jupyter notebook deploy/deploy.ipynb
   ```
   - 完整推理示例
   - 性能评估工具
   - 结果可视化

## ⚠️ 重要说明
1. **部署注意事项**
   - 必须使用 Jetson Nano 部署
   - PC 仅用于训练和转换
   - 确保环境版本匹配

2. **性能优化建议**
   - 使用 FP16 推理
   - 保持单批次输入
   - 避免频繁数据传输

3. **已知限制**
   - 输入分辨率固定
   - 仅支持单批次推理
   - 依赖特定版本组件

## 📚 参考实现
- [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/v6.0)：目标检测基础框架
- [RepVGG](https://github.com/DingXiaoH/RepVGG)：高效骨干网络
- [TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)：推理优化
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)：深度学习推理加速