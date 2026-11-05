"""
遥感影像 SegFormer 智能地物解译主程序 CLI (Remote Sensing SegFormer Land-Cover Runner)
用法示例:
    # 推荐：启动 SegFormer 多光谱物理融合全要素自然地物解译 (河流/森林/农田/城镇)
    python main.py --segformer

    # 水系专项高保真走向贴合提取与 QGIS 矢量导出
    python main.py --source data/SDC30_V003_50SMF_20210618.tif --target water

    # 自定义遥感影像解译
    python main.py --source data/sample_rs.tif --output outputs/
"""

import os
import sys
import argparse
from pathlib import Path

# 确保 UTF-8 标准输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from yolo_detector import RemoteSensingDetector
from rs_exporter import RemoteSensingExporter
from rs_segformer_engine import IndustrialSegFormerEngine, MultispectralSegFormerEngine, run_segformer_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="遥感影像 SegFormer 智能解译与全要素成果导出平台 (SegFormer + 多光谱融合)")
    parser.add_argument("--source", type=str, default="data/SDC30_V003_50SMF_20210618.tif", help="待解译遥感图像路径 (支持 .tif, .tiff, .png, .jpg)")
    parser.add_argument("--segformer", action="store_true", default=False, help="启动 SegFormer + 多光谱物理融合全要素解译引擎 (河流/植被/农田/城镇)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="通用辅助模型权重路径 (例如 yolo11n.pt, yolo11n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值 (默认: 0.45)")
    parser.add_argument("--slice", action="store_true", default=True, help="是否开启遥感大图切片滑窗推理 (适合超大分辨率影像)")
    parser.add_argument("--tile-size", type=int, default=640, help="切片滑窗窗口尺寸 (默认: 640)")
    parser.add_argument("--overlap", type=float, default=0.20, help="切片重叠率 (默认: 0.20)")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu 或 cuda:0)")
    parser.add_argument("--target", type=str, default="water", help="仅提取指定单一目标类别 (如: water/水系, road/道路, airplane/飞机, ship/船只 等, 默认: water)")
    parser.add_argument("--output", type=str, default="outputs", help="结果成果导出目录")
    return parser.parse_args()


def process_image(img_path: Path, detector: RemoteSensingDetector, exporter: RemoteSensingExporter, args):
    print(f"\n=======================================================")
    print(f"[>] 开始解译遥感影像: {img_path}")
    print(f"=======================================================")

    # 1. 读取遥感影像
    image_rgb, geo_meta = detector.load_image(str(img_path))
    h, w, c = image_rgb.shape
    print(f"[*] 影像尺寸: 宽 {w}px, 高 {h}px, 通道数 {c}")
    if geo_meta and geo_meta.get("crs"):
        print(f"[*] 空间参考 (CRS): {geo_meta['crs']}")
        print(f"[*] 地理范围 (Bounds): {geo_meta['bounds']}")

    # 特别处理：水系专项高精度识别引擎 (v3.1 高保真走向贴合)
    if args.target and args.target.strip().lower() in ["water", "水体", "水系"]:
        print(f"\n[*] 启动水系专项高精度识别引擎 (多光谱指数+抗阴影优化+河道高保真走向贴合)...")
        w_engine = IndustrialSegFormerEngine()
        w_res = w_engine.extract_water(str(img_path), min_area_pixels=30)
        w_summary = w_res["summary"]
        
        print("\n------------------ 水系解译结果统计 ------------------")
        print(f"[*] 提取水系斑块总数: {w_summary['total_water_bodies']} 个")
        print(f"[*] 水系覆盖总面积: {w_summary['total_water_km2']} km² (约 {w_summary['total_water_km2']*100:.1f} 公顷)")
        print(f"[*] 地表水体覆盖率: {w_summary['water_coverage_ratio_pct']} %")
        print(f"[*] 算法耗时: {w_summary['inference_time_ms']} ms")
        print("------------------------------------------------------")
        
        stem = f"{img_path.stem}_water"
        artifacts = w_engine.export_all_water_results(w_res, output_dir=args.output, prefix=stem)
        print("\n[OK] 水系专属解译成果已全部导出:")
        print(f"  1. 水系渲染标注图: {artifacts['annotated_image']}")
        print(f"  2. 二值水系掩膜 (Mask): {artifacts['water_mask']}")
        print(f"  3. 水系空间矢量 (GeoJSON): {artifacts['geojson']}")
        print(f"  4. 水系斑块属性表 (CSV): {artifacts['csv']}")
        print(f"  5. 水资源分析报告: {artifacts['report']}")
        return w_res

    # 特别处理：主干交通路网专项提取引擎
    if args.target and args.target.strip().lower() in ["road", "roads", "道路", "路网", "公路"]:
        print(f"\n[*] 启动主干交通路网专项提取引擎 (8方向线型结构元滤波+骨架细化)...")
        r_engine = IndustrialSegFormerEngine()
        r_res = r_engine.extract_roads(str(img_path), min_len_km=0.6)
        r_summary = r_res["summary"]
        
        print("\n------------------ 主干路网解译结果统计 ------------------")
        print(f"[*] 提取路网连续区段: {r_summary['total_roads']} 段")
        print(f"[*] 全区路网总里程数: {r_summary['total_length_km']} km")
        print(f"[*] 算法耗时: {r_summary['inference_time_ms']} ms")
        print("----------------------------------------------------------")
        
        stem = f"{img_path.stem}_road"
        artifacts = r_engine.export_all_road_results(r_res, output_dir=args.output, prefix=stem)
        print("\n[OK] 主干路网专属解译成果已全部导出:")
        print(f"  1. 路网渲染标注图: {artifacts['annotated_image']}")
        print(f"  2. 二值路网掩膜 (Mask): {artifacts['road_mask']}")
        print(f"  3. 路网空间矢量 (GeoJSON LineString): {artifacts['geojson']}")
        print(f"  4. 道路属性统计表 (CSV): {artifacts['csv']}")
        print(f"  5. 路网解译分析报告: {artifacts['report']}")
        return r_res

    # 2. 执行常规目标推理 (切片滑窗 或 整图直推)
    if args.target:
        print(f"[*] 启用单一目标专注识别模式: 仅提取 '{args.target}'，过滤其他无关类别...")

    if args.slice and (w > args.tile_size or h > args.tile_size):
        print(f"[*] 启用切片滑窗推理 (Tile: {args.tile_size}x{args.tile_size}, Overlap: {args.overlap*100}%)...")
        results = detector.detect_sliced(
            image=image_rgb,
            geo_meta=geo_meta,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap,
            target_class=args.target
        )
    else:
        print(f"[*] 采用整图直接推理模式...")
        results = detector.detect_single(image=image_rgb, geo_meta=geo_meta, target_class=args.target)

    # 3. 输出汇总统计
    summary = results.get("summary", {})
    print("\n------------------ 解译结果统计 ------------------")
    print(f"[*] 识别地物目标总数: {summary.get('total_objects', 0)} 个")
    print(f"[*] 模型推理用时: {summary.get('inference_time_ms', 0)} ms")
    print("[*] 各类地物目标数量分布:")
    for cls_name, count in summary.get("class_counts", {}).items():
        pct = summary.get("class_area_percentages", {}).get(cls_name, 0.0)
        print(f"    - {cls_name}: {count} 个 (占地面积约 {pct}%)")
    print("--------------------------------------------------")

    # 4. 导出多源成果
    stem = img_path.stem
    vis_path = exporter.draw_visual_results(image_rgb, results, f"{stem}_annotated.png")
    json_path = exporter.export_json(results, f"{stem}_detections.json")
    geojson_path = exporter.export_geojson(results, f"{stem}_detections.geojson")
    csv_path = exporter.export_csv(results, f"{stem}_detections.csv")
    chart_path = exporter.generate_chart(results, f"{stem}_summary_chart.png")
    report_path = exporter.generate_report(results, f"{stem}_report.md")

    print("\n[OK] 本次遥感识别全部成果已导出:")
    print(f"  1. 识别标注图: {vis_path}")
    print(f"  2. 空间矢量 (GeoJSON): {geojson_path}")
    print(f"  3. 结构化数据 (JSON): {json_path}")
    print(f"  4. 地表明细表 (CSV): {csv_path}")
    print(f"  5. 统计分析图: {chart_path}")
    print(f"  6. 分析报告: {report_path}")
    return results


def main():
    args = parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"[!] 输入影像不存在: {source_path}")
        sys.exit(1)

    # 优先分支：如果指定了 SegFormer 多光谱全要素解译引擎
    if args.segformer:
        run_segformer_pipeline(str(source_path), output_dir=args.output)
        return

    # 初始化识别器与导出器
    detector = RemoteSensingDetector(
        model_path=args.model,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device
    )
    exporter = RemoteSensingExporter(output_dir=args.output)

    # 处理单张或批量
    if source_path.is_file():
        process_image(source_path, detector, exporter, args)
    elif source_path.is_dir():
        supported_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        files = [p for p in source_path.iterdir() if p.suffix.lower() in supported_exts]
        print(f"[*] 发现 {len(files)} 张遥感影像待批处理...")
        for p in files:
            process_image(p, detector, exporter, args)


if __name__ == "__main__":
    main()
