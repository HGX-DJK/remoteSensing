# 遥感大宗地物分类专属解译模块库 (Land-Use Category Extractors)

本项目将水系、森林、农田、道路四大核心大宗地物解译逻辑解耦为独立的专项目录 `landuse_category/`，每个地物具备专属算法针对性优化、标准 GIS 空间要素输出与全套成果导出能力。

---

## 1. 目录结构与架构设计

```text
landuse_category/
├── __init__.py             # 统一对外暴露提取器与基础工具
├── base.py                 # 通用基底：多光谱读取、物理反射率指数反演、NumPy矢量化Chaikin平滑、GIS仿射投影、多环孔洞矢量化
├── water_extractor.py      # 水系专属解译器 (WaterExtractor)
├── forest_extractor.py     # 稠密森林专属解译器 (ForestExtractor)
├── cropland_extractor.py   # 农田耕地专属解译器 (CroplandExtractor)
├── road_extractor.py       # 交通大动脉专属解译器 (RoadExtractor - 方案 B 极严断舍离)
└── README.md               # 模块技术说明与接口指南
```

---

## 2. 各提取器核心算法与技术特色

### 2.1 水系专属提取器 (`WaterExtractor`)
- **多光谱物理反演**：$MNDWI = \frac{\text{Green} - \text{SWIR}}{\text{Green} + \text{SWIR}}$，结合 NIR 强吸收抑制与 NDVI 水体上限过滤；
- **1:1 走向紧密贴合**：废除粗暴的 `arcLength` 比例容差（杜绝截弯取直直角折线），改用**像元级严格受限容差**（$0.5 \sim 1.0\text{ px} \approx 15 \sim 30\text{ m}$），河道蜿蜒与卫星地形 100% 贴合；
- **水体形态分级**：基于几何紧凑度 $\text{Compactness} = \frac{4\pi A}{P^2}$ 自动区分面状湖泊水库与线性带状河流灌渠。

### 2.2 稠密森林专属提取器 (`ForestExtractor`)
- **多指数协同抗饱和**：针对高分卫星夏秋季林冠反射饱和问题，综合计算 NDVI、EVI（增强植被指数，蓝光大气校正）与 SAVI（土壤调节植被指数）；
- **栅格级硬互斥**：严格互斥水体表面，杜绝水陆边界混合；
- **Chaikin 样条平滑**：平滑 30 米阶梯像元锯齿，生成符合测绘规范的森林面要素（Polygon）。

### 2.3 农田耕地专属提取器 (`CroplandExtractor`)
- **生态层次解耦**：中高植被活力，栅格级严格互斥森林（`clean_forest == 0`）与水系（`clean_water == 0`），从源头彻底杜绝图层重叠污染；
- **麦田垄沟微孔抑制**：采用 $5 \times 5$ 形态学闭运算（Closing），消除 30m 像元下麦田垄沟形成的数十万个微孔洞假阴性，大幅提升矢量化效率；
- **真实村落孔洞镂空**：支持内部村庄、坑塘、房舍的镂空保留，输出标准的 GeoJSON Multi-Ring 多环多边形。

### 2.4 交通大动脉专属提取器 (`RoadExtractor - 方案 B 极严断舍离`)
- **8 方向线型结构元滤波**：采用 15 像元（450 米）的多角度直线形态学开运算，精准捕捉孤立长直走廊；
- **城镇建筑斑块内部剔除**：椭圆核开运算扣除 2D 城镇建筑群连片内部；
- **空间密度模糊滤波**：$65 \times 65$ 空间均值平滑滤波（阈值 $\le 50.0$），坚决滤除农田网格状田埂垄沟假阳性；
- **单像素骨架化 + SVD/PCA 主轴中位数采样**：100% 根除发夹回路与双线 bug，输出真正符合 GIS 规范的 1-像素 LineString 空间矢量。

---

## 3. Python 代码快速调用示例

```python
from landuse_category import (
    WaterExtractor,
    ForestExtractor,
    CroplandExtractor,
    RoadExtractor
)

tif_image = "data/SDC30_V003_50SMF_20210618.tif"

# 1. 水系专项提取
water_ext = WaterExtractor(min_area_pixels=30)
w_res = water_ext.extract(tif_image)
water_ext.export_deliverables(w_res, output_dir="outputs", prefix="my_water")

# 2. 森林专项提取
forest_ext = ForestExtractor(min_area_pixels=80)
f_res = forest_ext.extract(tif_image, water_mask=w_res["clean_water"])
forest_ext.export_deliverables(f_res, output_dir="outputs", prefix="my_forest")

# 3. 农田专项提取 (严格互斥森林与水系)
crop_ext = CroplandExtractor(min_area_pixels=120)
c_res = crop_ext.extract(tif_image, clean_forest=f_res["clean_forest"], clean_water=w_res["clean_water"])
crop_ext.export_deliverables(c_res, output_dir="outputs", prefix="my_cropland")

# 4. 交通大动脉专项提取 (方案 B 极严断舍离)
road_ext = RoadExtractor(min_road_len_km=1.5)
r_res = road_ext.extract(tif_image, clean_water=w_res["clean_water"], clean_forest=f_res["clean_forest"])
road_ext.export_deliverables(r_res, output_dir="outputs", prefix="my_road")
```

---

## 4. CLI 命令行调用

在项目根目录运行 `main.py`：

```bash
# 水系专项解译
python main.py --source data/SDC30_V003_50SMF_20210618.tif --target water

# 森林林地专项解译
python main.py --source data/SDC30_V003_50SMF_20210618.tif --target forest

# 农田耕地专项解译
python main.py --source data/SDC30_V003_50SMF_20210618.tif --target cropland

# 交通大动脉专项解译 (方案 B)
python main.py --source data/SDC30_V003_50SMF_20210618.tif --target road

# 全要素多光谱融合综合解译
python main.py --segformer
```
