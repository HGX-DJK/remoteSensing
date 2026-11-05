"""
遥感影像识别结果导出模块 (Remote Sensing Output Exporter)
支持可视化渲染图、GeoJSON、JSON、CSV、统计图表及检测报告的完整输出。
"""

import os
import tempfile
# 设置 matplotlib 缓存目录避免只读警告
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg") # 无头模式绘图
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from pathlib import Path

# 配置中文字体避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class RemoteSensingExporter:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_visual_results(
        self,
        image_rgb: np.ndarray,
        detection_result: Dict[str, Any],
        output_filename: str = "annotated_result.png",
        draw_masks: bool = True
    ) -> str:
        """
        绘制目标检测框、置信度标签及半透明分割掩膜
        """
        vis_img = image_rgb.copy()
        h, w, _ = vis_img.shape
        detections = detection_result.get("detections", [])

        # 1. 绘制分割多边形掩膜 (如果存在)
        if draw_masks:
            mask_overlay = vis_img.copy()
            has_masks = False
            for det in detections:
                poly = det.get("polygon")
                color = det.get("color", [0, 255, 0])
                if poly and len(poly) > 2:
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_overlay, [pts], color=color)
                    has_masks = True
            if has_masks:
                cv2.addWeighted(mask_overlay, 0.45, vis_img, 0.55, 0, vis_img)

        # 2. 绘制检测框与标签
        for det in detections:
            bbox = det.get("bbox", [])
            color = det.get("color", [0, 255, 0])
            # OpenCV使用BGR颜色
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # 绘制边框
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), bgr_color, 2)
                
                # 标签文本
                cls_zh = det.get("class_name_zh", "")
                cls_en = det.get("class_name", "")
                conf = det.get("confidence", 0.0)
                label_text = f"#{det['id']} {cls_zh}({cls_en}) {conf:.2f}"
                
                # 标签背景框
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(
                    vis_img,
                    (x1, max(0, y1 - text_h - 6)),
                    (x1 + text_w + 4, max(text_h + 6, y1)),
                    bgr_color,
                    -1
                )
                cv2.putText(
                    vis_img,
                    label_text,
                    (x1 + 2, max(text_h + 2, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        # 3. 叠加统计水印图例
        summary = detection_result.get("summary", {})
        total = summary.get("total_objects", len(detections))
        fps_info = f"Targets: {total} | Time: {summary.get('inference_time_ms', 0)}ms"
        cv2.rectangle(vis_img, (10, 10), (360, 45), (20, 20, 20), -1)
        cv2.putText(vis_img, "SegFormer Remote Sensing AI", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)
        cv2.putText(vis_img, fps_info, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        out_path = self.output_dir / output_filename
        # 转回BGR保存
        cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"[+] 识别可视化影像已保存至: {out_path}")
        return str(out_path)

    def export_json(self, detection_result: Dict[str, Any], filename: str = "detections.json") -> str:
        """导出全量 JSON 结果文件"""
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detection_result, f, ensure_ascii=False, indent=2)
        print(f"[+] 结构化 JSON 识别数据已保存至: {out_path}")
        return str(out_path)

    def export_geojson(self, detection_result: Dict[str, Any], filename: str = "detections.geojson") -> str:
        """
        导出符合 RFC 7946 标准的 GeoJSON 空间矢量文件
        可直接在 QGIS, ArcGIS, Cesium, Leaflet 中加载渲染
        """
        out_path = self.output_dir / filename
        features = []
        geo_meta = detection_result.get("geo_meta")
        detections = detection_result.get("detections", [])

        for det in detections:
            props = {
                "id": det["id"],
                "class_name": det["class_name"],
                "class_name_zh": det["class_name_zh"],
                "confidence": det["confidence"],
                "area_pixels": det["area_pixels"],
                "pixel_bbox": det["bbox"]
            }

            # 优先使用真实地理坐标
            if geo_meta and det.get("geo_bbox"):
                gb = det["geo_bbox"] # [minx, miny, maxx, maxy]
                # 多边形几何体
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [gb[0], gb[1]],
                        [gb[2], gb[1]],
                        [gb[2], gb[3]],
                        [gb[0], gb[3]],
                        [gb[0], gb[1]]
                    ]]
                }
            else:
                # 使用像素空间几何体
                bx = det["bbox"]
                geometry = {
                    "type": "Polygon",
                    "coordinates": [[
                        [bx[0], bx[1]],
                        [bx[2], bx[1]],
                        [bx[2], bx[3]],
                        [bx[0], bx[3]],
                        [bx[0], bx[1]]
                    ]]
                }

            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": geometry
            })

        geojson_doc = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": geo_meta.get("crs", "urn:ogc:def:crs:OGC:1.3:CRS84") if geo_meta else "PIXEL_SPACE"
                }
            },
            "features": features
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, ensure_ascii=False, indent=2)
        print(f"[+] 地理空间 GeoJSON 数据已保存至: {out_path}")
        return str(out_path)

    def export_csv(self, detection_result: Dict[str, Any], filename: str = "detections.csv") -> str:
        """导出地物检测明细表 CSV"""
        out_path = self.output_dir / filename
        detections = detection_result.get("detections", [])
        
        fieldnames = [
            "id", "class_name", "class_name_zh", "confidence",
            "xmin", "ymin", "xmax", "ymax", "area_pixels",
            "center_x_px", "center_y_px", "geo_lon", "geo_lat"
        ]

        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in detections:
                bbox = d.get("bbox", [0, 0, 0, 0])
                c_px = d.get("center_pixel", [0, 0])
                c_geo = d.get("center_geo") or ["-", "-"]
                writer.writerow({
                    "id": d["id"],
                    "class_name": d["class_name"],
                    "class_name_zh": d["class_name_zh"],
                    "confidence": d["confidence"],
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                    "area_pixels": d["area_pixels"],
                    "center_x_px": c_px[0],
                    "center_y_px": c_px[1],
                    "geo_lon": c_geo[0],
                    "geo_lat": c_geo[1]
                })
        print(f"[+] 地物数据统计表 CSV 已保存至: {out_path}")
        return str(out_path)

    def generate_chart(self, detection_result: Dict[str, Any], filename: str = "summary_chart.png") -> str:
        """生成地物识别统计分析图表"""
        out_path = self.output_dir / filename
        summary = detection_result.get("summary", {})
        counts = summary.get("class_counts", {})
        areas = summary.get("class_areas_pixel", {})

        if not counts:
            print("[*] 无检测目标，跳过图表生成。")
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        
        # 1. 柱状图：各类别地物数量
        labels = list(counts.keys())
        values = list(counts.values())
        colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(labels))))

        bars = ax1.bar(labels, values, color=colors[:len(labels)], edgecolor="black", alpha=0.85)
        ax1.set_title("遥感地物目标数量分布 (Target Counts)", fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel("数量 (个)", fontsize=10)
        ax1.tick_params(axis='x', rotation=25)
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # 2. 环形饼图：各类别地物面积占比
        area_labels = list(areas.keys())
        area_values = list(areas.values())
        
        ax2.pie(
            area_values,
            labels=area_labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(area_labels)],
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
        )
        ax2.set_title("地物覆盖面积比例 (Area Ratio)", fontsize=13, fontweight='bold', pad=12)

        plt.suptitle(f"遥感影像 SegFormer 地物智能解译统计 (总目标数: {summary.get('total_objects', 0)})", fontsize=15, y=0.98)
        plt.tight_layout()
        plt.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"[+] 统计分析图表已保存至: {out_path}")
        return str(out_path)

    def generate_report(self, detection_result: Dict[str, Any], filename: str = "report.md") -> str:
        """生成 Markdown 格式的专业检测分析报告"""
        out_path = self.output_dir / filename
        summary = detection_result.get("summary", {})
        counts = summary.get("class_counts", {})
        percentages = summary.get("class_area_percentages", {})
        geo_meta = detection_result.get("geo_meta")

        md_content = [
            "# 遥感影像 SegFormer 地物智能识别分析报告",
            "",
            "## 1. 基础信息概述",
            f"- **影像分辨率**: `{detection_result['image_shape'][1]} x {detection_result['image_shape'][0]}` 像素",
            f"- **通道数**: `{detection_result['image_shape'][2]}` 通道",
            f"- **推理耗时**: `{summary.get('inference_time_ms', 0)} ms`",
            f"- **地物目标总数**: `{summary.get('total_objects', 0)}` 个",
            f"- **空间坐标系 (CRS)**: `{geo_meta.get('crs') if geo_meta else '无投影元数据 (像素坐标系)'}`",
            "",
            "## 2. 地物类别统计清单",
            "| 序号 | 地物类别 | 识别数量 (个) | 面积占比 (%) | 像素总面积 (px²) |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        idx = 1
        for k in counts.keys():
            c_cnt = counts.get(k, 0)
            c_pct = percentages.get(k, 0.0)
            c_area = summary.get("class_areas_pixel", {}).get(k, 0.0)
            md_content.append(f"| {idx} | **{k}** | {c_cnt} | {c_pct}% | {c_area:,.1f} |")
            idx += 1

        md_content.extend([
            "",
            "## 3. 输出成果列表",
            "- 可视化标注影像: `annotated_result.png`",
            "- 空间矢量数据 (GeoJSON): `detections.geojson`",
            "- 结构化数据 (JSON): `detections.json`",
            "- 地物明细数据表 (CSV): `detections.csv`",
            "- 统计图表 (Chart): `summary_chart.png`",
            ""
        ])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        print(f"[+] 解译分析报告已保存至: {out_path}")
        return str(out_path)
