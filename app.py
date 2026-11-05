"""
遥感影像 SegFormer 地物智能解译 Web 可视化平台 (FastAPI 后端)
启动方式:
    python app.py
访问地址:
    http://localhost:8000
"""

import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 确保 UTF-8 输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from yolo_detector import RemoteSensingDetector
from rs_exporter import RemoteSensingExporter
from rs_segformer_engine import IndustrialSegFormerEngine

app = FastAPI(title="Remote Sensing SegFormer Land-Cover AI", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 挂载静态文件目录
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 缓存检测器实例 (避免重复加载)
detector_cache = {}


def get_detector(model_name: str, conf: float, iou: float, device: str = "cpu") -> RemoteSensingDetector:
    key = f"{model_name}_{conf}_{iou}_{device}"
    if key not in detector_cache:
        model_path = model_name if os.path.exists(model_name) else f"{model_name}.pt"
        if not os.path.exists(model_path):
            model_path = "yolo11n.pt" # 默认回退
        detector_cache[key] = RemoteSensingDetector(
            model_path=model_path,
            conf_thres=conf,
            iou_thres=iou,
            device=device
        )
    return detector_cache[key]


@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "遥感影像 SegFormer 地物识别 API 服务已启动"}


@app.post("/api/detect")
async def detect_remote_sensing(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None),
    model_name: str = Form("yolo11n.pt"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    slice_inference: bool = Form(True),
    tile_size: int = Form(640),
    overlap: float = Form(0.20),
    target: Optional[str] = Form(None),
):
    try:
        req_id = uuid.uuid4().hex[:8]
        
        # 1. 保存上传文件或使用样例
        if file and file.filename:
            file_ext = Path(file.filename).suffix.lower()
            saved_name = f"upload_{req_id}{file_ext}"
            file_path = UPLOAD_DIR / saved_name
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif sample_name:
            file_path = Path("data") / sample_name
            if not file_path.exists():
                file_path = Path(sample_name)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="指定的样例文件不存在")
        else:
            file_path = Path("data/sample_rs.tif")
            if not file_path.exists():
                from generate_sample_data import prepare_sample_dataset
                prepare_sample_dataset()

        # 2. 加载识别模型
        detector = get_detector(model_name, conf, iou)
        exporter = RemoteSensingExporter(output_dir=str(OUTPUT_DIR))

        # 3. 读取图像与地理元数据
        image_rgb, geo_meta = detector.load_image(str(file_path))
        h, w, c = image_rgb.shape

        # 4. 执行推理
        target_cls = target.strip() if target and target.strip() != "all" else None

        # 如果用户指定提取水系 (启动水系专项高精度引擎)
        if target_cls in ["water", "水体", "水系"]:
            w_engine = IndustrialSegFormerEngine()
            w_res = w_engine.extract_water(str(file_path), min_area_pixels=30)
            stem = f"water_{req_id}"
            art_paths = w_engine.export_all_water_results(w_res, output_dir=str(OUTPUT_DIR), prefix=stem)
            
            # 格式化检测目标列表
            det_items = []
            for wp in w_res["water_polygons"]:
                det_items.append({
                    "id": wp["id"],
                    "class_name": "water",
                    "class_name_zh": f"水系({wp['type']})",
                    "confidence": wp["confidence"],
                    "color": [30, 144, 255],
                    "bbox": wp["bbox_pixel"],
                    "center_pixel": wp["center_pixel"],
                    "center_geo": wp["center_geo"],
                    "area_pixels": wp["area_pixels"]
                })
            
            summary_info = {
                "total_objects": w_res["summary"]["total_water_bodies"],
                "class_counts": {"水系": w_res["summary"]["total_water_bodies"]},
                "class_areas_pixel": {"水系": w_res["summary"]["total_water_pixels"]},
                "class_area_percentages": {"水系": w_res["summary"]["water_coverage_ratio_pct"]},
                "inference_time_ms": w_res["summary"]["inference_time_ms"]
            }

            return {
                "success": True,
                "request_id": req_id,
                "image_info": {
                    "filename": file_path.name,
                    "width": w_res["image_shape"][1],
                    "height": w_res["image_shape"][0],
                    "channels": w_res["image_shape"][2],
                    "has_georeference": w_res["geo_meta"] is not None,
                    "crs": w_res["geo_meta"].get("crs") if w_res["geo_meta"] else None
                },
                "summary": summary_info,
                "detections": det_items,
                "artifacts": {
                    "annotated_image": f"/outputs/{Path(art_paths['annotated_image']).name}",
                    "geojson": f"/outputs/{Path(art_paths['geojson']).name}",
                    "json": f"/outputs/{Path(art_paths['geojson']).name}",
                    "csv": f"/outputs/{Path(art_paths['csv']).name}",
                    "chart": f"/outputs/{Path(art_paths['annotated_image']).name}",
                    "report": f"/outputs/{Path(art_paths['report']).name}"
                }
            }

        # 4. 执行通用目标推理
        if slice_inference and (w > tile_size or h > tile_size):
            results = detector.detect_sliced(
                image=image_rgb,
                geo_meta=geo_meta,
                tile_size=tile_size,
                overlap_ratio=overlap,
                target_class=target_cls
            )
        else:
            results = detector.detect_single(image=image_rgb, geo_meta=geo_meta, target_class=target_cls)

        # 5. 导出各类成果
        stem = f"rs_{req_id}"
        vis_filename = f"{stem}_annotated.png"
        json_filename = f"{stem}_detections.json"
        geojson_filename = f"{stem}_detections.geojson"
        csv_filename = f"{stem}_detections.csv"
        chart_filename = f"{stem}_chart.png"
        report_filename = f"{stem}_report.md"

        exporter.draw_visual_results(image_rgb, results, vis_filename)
        exporter.export_json(results, json_filename)
        exporter.export_geojson(results, geojson_filename)
        exporter.export_csv(results, csv_filename)
        exporter.generate_chart(results, chart_filename)
        exporter.generate_report(results, report_filename)

        # 返回 JSON 响应
        return {
            "success": True,
            "request_id": req_id,
            "image_info": {
                "filename": file_path.name,
                "width": w,
                "height": h,
                "channels": c,
                "has_georeference": geo_meta is not None and geo_meta.get("crs") is not None,
                "crs": geo_meta.get("crs") if geo_meta else None,
                "bounds": geo_meta.get("bounds") if geo_meta else None
            },
            "summary": results.get("summary", {}),
            "detections": results.get("detections", []),
            "artifacts": {
                "annotated_image": f"/outputs/{vis_filename}",
                "geojson": f"/outputs/{geojson_filename}",
                "json": f"/outputs/{json_filename}",
                "csv": f"/outputs/{csv_filename}",
                "chart": f"/outputs/{chart_filename}",
                "report": f"/outputs/{report_filename}",
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/samples")
def list_samples():
    """获取系统内置及工作区的遥感样例影像"""
    supported = {".tif", ".tiff", ".png", ".jpg"}
    samples = []
    
    # 扫描 data 目录
    data_dir = Path("data")
    if data_dir.exists():
        for f in data_dir.iterdir():
            if f.suffix.lower() in supported:
                samples.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                })
    # 扫描工作区根目录
    for f in Path(".").glob("*.tif*"):
        if f.is_file() and not f.name.startswith("."):
            samples.append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })
    return {"samples": samples}


if __name__ == "__main__":
    print("[*] 正在启动遥感影像 SegFormer 地物智能解译 Web 服务...")
    print("[*] 访问地址: http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
