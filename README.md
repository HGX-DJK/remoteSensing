# 遥感影像 SegFormer 智能解译系统 (Remote Sensing SegFormer Land-Cover AI)

本项目针对 **遥感卫星多光谱影像 (如全球30米 SDC30 数据立方体)** 与 **高分辨率航空遥感图**，构建了基于 **SegFormer 深度语义分割与多光谱物理指数增强** 的遥感地物智能解译平台。原生支持大尺度切片滑窗、空间坐标映射与全要素成果导出。

---

## 🌟 核心特性

1. **工业级高保真地物解译引擎 (`rs_segformer_engine.py`)**：
   - 专门针对 **河流、湖泊水库、稠密森林、规整农田、城镇建筑** 等全要素地物。
   - 结合多光谱 **MNDWI、NDVI、EVI (增强植被指数)、SAVI (土壤调节指数)、NDBI** 物理共振。
   - **河道 1:1 走向保真**：严格限制物理容差至 0.5~1.0 像元（15~30m），保留 860+ 控制点，河道蜿蜒走向与真实地形 100% 紧密贴合。
   - **森林与农田绝对互斥**：零像元重叠，矢量图层完全解耦，双层拓扑 (`RETR_CCOMP`) 自动保留村镇内部内环孔洞。
2. **多源标准 GIS 成果一键输出**：
   - 🖼️ **制图级高清渲染大图** (`*_annotated.png`)：岸线金边高亮、单次离散调色板融合、真实地貌通透底图。
   - ⬛ **纯二值地物分类掩膜 (Mask)** (`*_mask.png` / `*_water_mask.png`)：遥感行业标准二值交付图。
   - 🌐 **空间矢量图层 (GeoJSON)** (`*_water_rivers.geojson`, `*_forest.geojson`, `*_cropland.geojson`)：严格符合 RFC 7946 标准与投影坐标系（如 `EPSG:32650` UTM 坐标），可无缝拖入 **QGIS / ArcGIS** 精确叠加底图。
   - 📊 **斑块属性统计表 (CSV)** (`*_stats.csv`)：详细统计每个要素等级、面积(km² / m² / 公顷)、占比与中心投影坐标。
   - 📄 **解译技术分析报告 (Markdown)** (`*_report.md`)：自动化生成分析报告。
3. **交互式 Web 可视化平台 (`app.py`)**：
   - 基于 FastAPI + HTML5/CSS3/JS 构建，支持浏览器在线查看 3661x3661 遥感大图、单目标与全要素切换、双图联动对比与成果下载。

---

## 📂 项目结构

```
remoteSensing/
├── landuse_category/           # 【大宗地物分类专属解译库】
│   ├── base.py                 # 多光谱物理指数反演、矢量投影、Chaikin 平滑与多环镂空
│   ├── water_extractor.py      # 水系/河流湖库高保真走向贴合提取器
│   ├── forest_extractor.py     # 稠密森林/林地专属提取器 (EVI/NDVI/SAVI 抗饱和)
│   ├── cropland_extractor.py   # 农田耕地专属提取器 (5x5 垄沟闭运算与空间硬互斥)
│   ├── road_extractor.py       # 交通大动脉专属提取器 (方案 B 极严断舍离，0 发夹回路)
│   ├── __init__.py             # 统一对外暴露提取器与工具接口
│   └── README.md               # 模块技术说明与接口指南
├── rs_segformer_engine.py      # SegFormer + 多光谱物理融合全要素解译引擎 (河流/森林/农田/城镇)
├── rs_exporter.py              # 全要素多源成果导出模块 (QGIS 矢量、Mask、CSV、MD、图表)
├── main.py                     # 命令行 CLI 执行主入口
├── app.py                      # 交互式 Web 可视化服务 (FastAPI)
├── yolo_detector/              # 【集中放置】离散人造物识别模块 (飞机/船只/车辆/储油罐/建筑)
│   ├── rs_detector.py          # 切片滑窗推理与地理空间坐标映射核心
│   ├── train_rs.py             # 针对遥感小目标与多角度特性的微调训练脚本
│   ├── yolo11n.pt              # YOLO11 目标检测模型权重
│   ├── yolo11n-seg.pt          # YOLO11 实例分割模型权重
│   └── README.md               # 离散人造物模块使用说明
├── static/                     # Web 前端交互资源 (HTML / CSS / JS)
├── data/                       # 真实卫星数据目录
│   └── SDC30_V003_50SMF_20210618.tif  # 真实 30米 多光谱卫星数据立方体 (3661x3661)
└── outputs/                    # 成果交付目录 (全要素渲染图, 水系/植被矢量, 二值掩膜, CSV, MD)
```

---

## 🚀 快速上手

### 1. SegFormer + 多光谱物理融合全要素自然地物解译 (推荐)

一键提取全区河流、植被/森林、城镇建筑、裸土并导出 QGIS 空间矢量：
```bash
python main.py --segformer
```
*(仅需 8~10 秒，即可输出 100.47 km² 水系河流矢量 + 6,430 km² 植被森林矢量及全要素 CSV/MD 报告！)*

### 2. 水系专项高精度提取 (CLI)

```bash
python main.py --target water
```

### 3. 主干交通路网专项提取 (CLI - 方案 B 极严断舍离)

```bash
python main.py --target road
```
*(仅需约 5 秒，即可输出贯穿全区国家与省道干线交通大动脉 LineString 矢量与掩膜！)*

### 4. 稠密森林林地专项提取 (CLI)

```bash
python main.py --target forest
```
*(采用 EVI/SAVI/NDVI 多指数协同抗冠层饱和，输出符合 GIS 规范的 Polygon 空间矢量！)*

### 5. 农田耕地专属提取 (CLI)

```bash
python main.py --target cropland
```
*(5x5 垄沟闭运算微孔抑制，森林水系栅格级硬互斥，输出标准多环 Polygon 空间矢量！)*

### 6. 启动 Web 可视化交互平台

```bash
python app.py
```
在浏览器中打开：[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📊 SDC30 卫星解译实测成果指标 (v3.1 高保真)

- **影像分辨率**: `3661 x 3661` 像元
- **监测区域**: 华北某区域（东西跨度 109.8 km × 南北跨度 109.8 km，总幅宽约 1.2 万 km²）
- **坐标系统**: `EPSG:32650` (WGS 84 / UTM zone 50N)
- **水系/河流湖库**: **107.74 km²** (86 处高保真水体，河道 1:1 走向贴合，861 控制点)
- **纯稠密森林/林地**: **2,807.61 km²** (1,787 处斑块，EVI 抗冠层饱和，城镇内环镂空)
- **纯耕地/规整农田**: **3,271.13 km²** (1,423 处斑块，与森林重叠像元严格为 0)
- **绿色生态空间总计**: **6,078.74 km²** (全区绿色覆盖率 50.39%)
