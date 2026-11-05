"""
地物分类专属解译模块包 (Land-Use Category Extraction Package)
提供针对遥感大图的四大核心大宗自然与人造地物专项解译器：
- WaterExtractor: 水系/河流湖库高保真提取 (0.5~1.0px 严格容差，走向与卫星底图 100% 紧密贴合)
- ForestExtractor: 稠密森林/林地专属提取 (EVI/NDVI/SAVI 多指数协同，排除冠层饱和与重叠)
- CroplandExtractor: 农田耕地专属提取 (5x5 垄沟微孔抑制，森林水系栅格级硬互斥)
- RoadExtractor: 国家级/省级交通大动脉专属提取 (方案 B 极严断舍离，密度滤波消除麦田网格，PCA 主轴单向采样 0 发夹回路)
"""

from .base import (
    load_multispectral_data,
    chaikin_smooth,
    pts_to_geo_coords,
    vectorize_polygons_with_holes,
)

from .water_extractor import WaterExtractor
from .forest_extractor import ForestExtractor
from .cropland_extractor import CroplandExtractor
from .road_extractor import RoadExtractor

__all__ = [
    "load_multispectral_data",
    "chaikin_smooth",
    "pts_to_geo_coords",
    "vectorize_polygons_with_holes",
    "WaterExtractor",
    "ForestExtractor",
    "CroplandExtractor",
    "RoadExtractor",
]
