"""
遥感影像地物识别与切片推理模块 (Remote Sensing Feature Detector)
支持遥感大图切片滑窗推理 (Sliced / Sliding Window Inference)、GeoTIFF 地理坐标映射、
目标检测 (Object Detection) 与 实例分割 (Instance Segmentation)。
"""

import os
import time
import yaml
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from ultralytics import YOLO

# 尝试导入遥感地理信息读取库 rasterio
try:
    import rasterio
    proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    if os.path.exists(proj_dir):
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"] = proj_dir
    from rasterio.transform import xy as rasterio_xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class RemoteSensingDetector:
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        config_path: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        device: str = "cpu"
    ):
        """
        初始化遥感影像地物识别器
        :param model_path: YOLO权重文件路径 (例如 yolo11n.pt, yolo11n-seg.pt 或训练好的遥感专用权重)
        :param config_path: 类别配置文件路径 (configs/rs_classes.yaml)
        :param conf_thres: 置信度阈值
        :param iou_thres: NMS IoU阈值
        :param device: 运行设备 ('cpu' 或 '0' / 'cuda')
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        
        # 加载YOLO模型
        print(f"[*] 正在加载 YOLO 识别模型: {model_path} (设备: {device})...")
        self.model = YOLO(model_path)
        self.is_segmentation = "seg" in str(model_path).lower()
        
        # 加载类别配置
        self.categories = {}
        self.coco_to_rs_map = {}
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str]):
        """加载类别字典与色彩映射配置"""
        default_config = Path(__file__).parent / "configs" / "rs_classes.yaml"
        target_path = Path(config_path) if config_path else default_config
        
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.categories = cfg.get("categories", {})
                self.coco_to_rs_map = cfg.get("coco_to_rs_mapping", {})
        else:
            # 默认类别回退
            self.categories = {
                0: {"name": "airplane", "zh": "飞机", "color": [255, 56, 56]},
                1: {"name": "ship", "zh": "船只", "color": [56, 156, 255]},
                2: {"name": "storage_tank", "zh": "储罐", "color": [255, 159, 28]},
                3: {"name": "building", "zh": "建筑物", "color": [255, 222, 89]},
                4: {"name": "vehicle", "zh": "车辆", "color": [0, 204, 150]},
                5: {"name": "water", "zh": "水体", "color": [31, 119, 180]},
                6: {"name": "vegetation", "zh": "植被", "color": [44, 160, 44]},
                7: {"name": "road", "zh": "道路", "color": [127, 127, 127]}
            }

    def load_image(self, image_path: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        读取遥感影像，支持常规格式(PNG, JPG)及遥感GeoTIFF(.tif, .tiff)
        返回: (RGB图像uint8三通道, 地理元数据字典)
        """
        path = str(image_path)
        geo_meta = None
        ext = os.path.splitext(path)[1].lower()

        if ext in [".tif", ".tiff"] and HAS_RASTERIO:
            try:
                with rasterio.open(path) as src:
                    # 获取地理元数据
                    geo_meta = {
                        "crs": str(src.crs) if src.crs else None,
                        "transform": src.transform,
                        "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
                        "width": src.width,
                        "height": src.height
                    }
                    # 读取通道
                    count = src.count
                    if count >= 3:
                        # 对于多光谱卫星影像 (如 6 波段 Landsat/Sentinel)，如果波段数>=4，通常 4,3,2 为真彩色(红绿蓝)，否则取 1,2,3
                        band_indices = [4, 3, 2] if count >= 4 else [1, 2, 3]
                        try:
                            img_arr = src.read(band_indices)
                        except Exception:
                            img_arr = src.read([1, 2, 3])
                        img_arr = np.transpose(img_arr, (1, 2, 0)) # 转为 (H, W, C)
                    else:
                        single_band = src.read(1)
                        img_arr = cv2.cvtColor(single_band, cv2.COLOR_GRAY2RGB)

                    # 像素快速归一化为 0-255 uint8 (针对超大图采用空间步长采样加速百分位数计算)
                    if img_arr.dtype != np.uint8:
                        sample = img_arr[::8, ::8] if img_arr.shape[0] > 1000 else img_arr
                        p_min, p_max = np.percentile(sample, (2, 98))
                        if p_max > p_min:
                            img_arr = np.clip(img_arr, p_min, p_max)
                            img_arr = ((img_arr - p_min) / (p_max - p_min) * 255).astype(np.uint8)
                        else:
                            img_arr = np.zeros_like(img_arr, dtype=np.uint8)
                    return img_arr, geo_meta
            except Exception as e:
                print(f"[!] rasterio 读取失败，尝试使用 OpenCV: {e}")

        # 使用 OpenCV 读取普通图像或回退读取
        img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取影像文件: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, geo_meta

    def _pixel_to_geo(self, px: float, py: float, geo_meta: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
        """将像素坐标转换为地理坐标 (经度/投影X, 纬度/投影Y)"""
        if geo_meta and HAS_RASTERIO and geo_meta.get("transform"):
            x_geo, y_geo = rasterio_xy(geo_meta["transform"], py, px)
            return round(x_geo, 6), round(y_geo, 6)
        return None

    def _resolve_class_info(self, raw_cls_name: str) -> Tuple[str, str, List[int]]:
        """将YOLO模型输出类别解析为遥感地物类别名称、中文名称及绘制颜色"""
        rs_name = self.coco_to_rs_map.get(raw_cls_name, raw_cls_name)
        
        for _, cat in self.categories.items():
            if cat.get("name") == rs_name or cat.get("name") == raw_cls_name:
                return cat.get("name", rs_name), cat.get("zh", rs_name), cat.get("color", [0, 255, 0])
        
        # 未配置类别的色彩哈希分配
        hash_val = abs(hash(raw_cls_name))
        color = [(hash_val * 37) % 255, (hash_val * 67) % 255, (hash_val * 97) % 255]
        return rs_name, rs_name, color

    def detect_single(
        self,
        image: np.ndarray,
        geo_meta: Optional[Dict[str, Any]] = None,
        target_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        直接整图推理 (适合常规尺寸遥感图)
        :param target_class: 指定仅识别某一特定类别 (如 "airplane", "飞机", "ship" 等)，其余类别自动过滤
        """
        start_time = time.time()
        h, w, c = image.shape
        
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        names = results.names
        
        boxes = results.boxes
        masks = results.masks if self.is_segmentation and results.masks is not None else None

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy()

            det_counter = 1
            for i in range(len(boxes)):
                raw_cls = names[int(cls_list[i])]
                conf = float(conf_list[i])
                bbox = [round(float(v), 2) for v in xyxy_list[i]]
                x1, y1, x2, y2 = bbox
                
                cls_name, cls_zh, color = self._resolve_class_info(raw_cls)
                
                # 单目标筛选过滤
                if target_class:
                    tgt = target_class.strip().lower()
                    if tgt not in [cls_name.lower(), cls_zh.lower(), raw_cls.lower()]:
                        continue
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = round((x2 - x1) * (y2 - y1), 2)

                # 地理坐标换算
                geo_center = self._pixel_to_geo(cx, cy, geo_meta)
                geo_bbox = None
                if geo_meta:
                    p1_geo = self._pixel_to_geo(x1, y1, geo_meta)
                    p2_geo = self._pixel_to_geo(x2, y2, geo_meta)
                    if p1_geo and p2_geo:
                        geo_bbox = [p1_geo[0], p2_geo[1], p2_geo[0], p1_geo[1]]

                # 分割多边形
                polygon = None
                if masks is not None and i < len(masks.xy):
                    poly_pts = masks.xy[i]
                    if len(poly_pts) > 0:
                        polygon = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in poly_pts]

                detections.append({
                    "id": det_counter,
                    "class_name": cls_name,
                    "class_name_zh": cls_zh,
                    "confidence": round(conf, 4),
                    "color": color,
                    "bbox": bbox,
                    "center_pixel": [round(cx, 2), round(cy, 2)],
                    "center_geo": geo_center,
                    "geo_bbox": geo_bbox,
                    "area_pixels": area,
                    "polygon": polygon
                })
                det_counter += 1

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = self._build_summary(detections, duration_ms, (h, w))

        return {
            "image_shape": [h, w, c],
            "geo_meta": geo_meta,
            "summary": summary,
            "detections": detections
        }

    def detect_sliced(
        self,
        image: np.ndarray,
        geo_meta: Optional[Dict[str, Any]] = None,
        tile_size: int = 640,
        overlap_ratio: float = 0.2,
        target_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        大尺寸遥感影像切片滑窗推理 (Sliced Inference)
        :param target_class: 指定仅识别某一特定类别 (如 "airplane", "飞机", "ship" 等)，其余类别自动过滤
        """
        start_time = time.time()
        h, w, c = image.shape
        stride = int(tile_size * (1 - overlap_ratio))
        
        # 生成切片坐标
        x_starts = list(range(0, max(1, w - tile_size + 1), stride))
        if len(x_starts) == 0 or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))
            
        y_starts = list(range(0, max(1, h - tile_size + 1), stride))
        if len(y_starts) == 0 or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))
            
        all_boxes = []
        all_confs = []
        all_raw_classes = []
        all_polygons = []

        total_tiles = len(x_starts) * len(y_starts)
        print(f"[*] 遥感影像尺寸: {w}x{h}，切片大小: {tile_size}x{tile_size}，总切片数: {total_tiles}...")

        # 逐切片推理
        for y0 in y_starts:
            for x0 in x_starts:
                x1 = min(w, x0 + tile_size)
                y1 = min(h, y0 + tile_size)
                tile = image[y0:y1, x0:x1]

                results = self.model.predict(
                    source=tile,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    device=self.device,
                    verbose=False
                )[0]

                boxes = results.boxes
                masks = results.masks if self.is_segmentation and results.masks is not None else None
                names = results.names

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy()

                    for bi in range(len(boxes)):
                        # 将切片局部坐标平移映射到整图全局坐标
                        gx1 = float(xyxy[bi][0] + x0)
                        gy1 = float(xyxy[bi][1] + y0)
                        gx2 = float(xyxy[bi][2] + x0)
                        gy2 = float(xyxy[bi][3] + y0)
                        
                        all_boxes.append([gx1, gy1, gx2, gy2])
                        all_confs.append(float(confs[bi]))
                        all_raw_classes.append(names[int(classes[bi])])

                        poly = None
                        if masks is not None and bi < len(masks.xy):
                            m_xy = masks.xy[bi]
                            if len(m_xy) > 0:
                                poly = [[float(pt[0] + x0), float(pt[1] + y0)] for pt in m_xy]
                        all_polygons.append(poly)

        # 全局非极大值抑制 (NMS) 合并跨边界重叠框
        final_detections = []
        if len(all_boxes) > 0:
            boxes_np = np.array(all_boxes, dtype=np.float32)
            confs_np = np.array(all_confs, dtype=np.float32)

            # 按类别分组做 NMS
            unique_classes = set(all_raw_classes)
            keep_indices = []

            for cls_item in unique_classes:
                cls_idxes = [i for i, c in enumerate(all_raw_classes) if c == cls_item]
                cls_boxes = boxes_np[cls_idxes]
                cls_confs = confs_np[cls_idxes]
                
                # cv2.dnn.NMSBoxes 接收 (x, y, w, h)
                cv_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in cls_boxes]
                indices = cv2.dnn.NMSBoxes(
                    bboxes=cv_boxes,
                    scores=cls_confs.tolist(),
                    score_threshold=self.conf_thres,
                    nms_threshold=self.iou_thres
                )
                
                if len(indices) > 0:
                    for idx in indices.flatten():
                        keep_indices.append(cls_idxes[idx])

            # 整理最终识别结果
            det_id = 1
            for k in keep_indices:
                raw_cls = all_raw_classes[k]
                cls_name, cls_zh, color = self._resolve_class_info(raw_cls)
                
                # 单目标筛选过滤
                if target_class:
                    tgt = target_class.strip().lower()
                    if tgt not in [cls_name.lower(), cls_zh.lower(), raw_cls.lower()]:
                        continue
                bbox = [round(v, 2) for v in all_boxes[k]]
                bx1, by1, bx2, by2 = bbox
                conf = round(all_confs[k], 4)
                cx = (bx1 + bx2) / 2.0
                cy = (by1 + by2) / 2.0
                area = round((bx2 - bx1) * (by2 - by1), 2)

                geo_center = self._pixel_to_geo(cx, cy, geo_meta)
                geo_bbox = None
                if geo_meta:
                    p1_geo = self._pixel_to_geo(bx1, by1, geo_meta)
                    p2_geo = self._pixel_to_geo(bx2, by2, geo_meta)
                    if p1_geo and p2_geo:
                        geo_bbox = [p1_geo[0], p2_geo[1], p2_geo[0], p1_geo[1]]

                poly = None
                if all_polygons[k] is not None:
                    poly = [[round(p[0], 2), round(p[1], 2)] for p in all_polygons[k]]

                final_detections.append({
                    "id": det_id,
                    "class_name": cls_name,
                    "class_name_zh": cls_zh,
                    "confidence": conf,
                    "color": color,
                    "bbox": bbox,
                    "center_pixel": [round(cx, 2), round(cy, 2)],
                    "center_geo": geo_center,
                    "geo_bbox": geo_bbox,
                    "area_pixels": area,
                    "polygon": poly
                })
                det_id += 1

        duration_ms = round((time.time() - start_time) * 1000, 2)
        summary = self._build_summary(final_detections, duration_ms, (h, w))

        return {
            "image_shape": [h, w, c],
            "geo_meta": geo_meta,
            "summary": summary,
            "detections": final_detections
        }

    def _build_summary(self, detections: List[Dict[str, Any]], duration_ms: float, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """统计地物分布指标"""
        h, w = img_shape
        total_pixels = float(h * w)
        counts = {}
        areas = {}

        for d in detections:
            c = d["class_name_zh"]
            counts[c] = counts.get(c, 0) + 1
            areas[c] = round(areas.get(c, 0.0) + d["area_pixels"], 2)

        area_percentages = {
            k: round((v / total_pixels) * 100, 2) for k, v in areas.items()
        }

        return {
            "total_objects": len(detections),
            "class_counts": counts,
            "class_areas_pixel": areas,
            "class_area_percentages": area_percentages,
            "inference_time_ms": duration_ms
        }
