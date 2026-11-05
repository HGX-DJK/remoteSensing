# 遥感影像离散人造物识别模块 (YOLO Discrete Objects Detector)

本目录集中存放针对 **离散人造地物目标**（如飞机、船舶、储油罐、车辆、独立建筑物等）的 YOLO 检测与实例分割组件。

---

## 📂 模块结构

- `rs_detector.py`: 遥感大图切片滑窗 (Sliced Window) 推理与地理坐标投影映射模块
- `train_rs.py`: 针对遥感多角度与小目标特性的 YOLO 模型训练与微调脚本 (支持 DOTA、NWPU VHR-10 数据集格式)
- `yolo11n.pt`: YOLO11 目标检测基础模型权重
- `yolo11n-seg.pt`: YOLO11 实例分割模型权重

---

## 🚀 独立使用示例

```python
from yolo_detector import RemoteSensingDetector

# 初始化检测器
detector = RemoteSensingDetector(model_path="yolo11n.pt")

# 读取并切片推理
image_rgb, geo_meta = detector.load_image("data/sample_rs.tif")
results = detector.detect_sliced(image_rgb, geo_meta=geo_meta, tile_size=640, overlap_ratio=0.2)
```
