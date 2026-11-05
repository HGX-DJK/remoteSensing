"""
遥感影像离散人造物识别与分割模块 (Remote Sensing YOLO Discrete Objects Module)
包含：飞机、船舶、储油罐、车辆、建筑物等离散人造目标的检测与实例分割
"""

from .rs_detector import RemoteSensingDetector

__all__ = ["RemoteSensingDetector"]
