// 遥感影像解译 Web 客户端脚本
let currentChart = null;
let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
  initUI();
  initChart();
});

function initUI() {
  const confThres = document.getElementById("confThres");
  const confVal = document.getElementById("confVal");
  confThres.addEventListener("input", (e) => confVal.textContent = e.target.value);

  const iouThres = document.getElementById("iouThres");
  const iouVal = document.getElementById("iouVal");
  iouThres.addEventListener("input", (e) => iouVal.textContent = e.target.value);

  const sliceToggle = document.getElementById("sliceToggle");
  const sliceParams = document.getElementById("sliceParams");
  sliceToggle.addEventListener("change", (e) => {
    sliceParams.style.display = e.target.checked ? "block" : "none";
  });

  // 拖拽与点击上传
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const selectedBadge = document.getElementById("selectedFileName");

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    selectedBadge.style.display = "block";
    selectedBadge.textContent = `已选文件: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    // 原始预览
    const reader = new FileReader();
    reader.onload = (ev) => {
      if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
        document.getElementById("imgOriginal").src = ev.target.result;
      }
    };
    reader.readAsDataURL(file);
  }

  // 样例切换
  const sampleSelect = document.getElementById("sampleSelect");
  sampleSelect.addEventListener("change", (e) => {
    selectedFile = null;
    selectedBadge.style.display = "none";
  });

  // 视图切换
  const tabBtns = document.querySelectorAll(".tab-btn");
  const cardOriginal = document.getElementById("cardOriginal");
  const cardResult = document.getElementById("cardResult");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.dataset.view;

      if (mode === "split") {
        cardOriginal.style.display = "block";
        cardResult.style.display = "block";
      } else if (mode === "result") {
        cardOriginal.style.display = "none";
        cardResult.style.display = "block";
      } else if (mode === "original") {
        cardOriginal.style.display = "block";
        cardResult.style.display = "none";
      }
    });
  });

  // 运行解译按钮
  document.getElementById("runBtn").addEventListener("click", runDetection);
}

function initChart() {
  const ctx = document.getElementById("categoryChart").getContext("2d");
  currentChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["水系水域 (0.75%)", "陆地/植被/其他 (99.25%)"],
      datasets: [{
        data: [0.75, 99.25],
        backgroundColor: ["#1e90ff", "#2ea043"],
        borderWidth: 2,
        borderColor: "#161b22"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#8b949e", font: { size: 10 } }
        }
      }
    }
  });
}

async function runDetection() {
  const runBtn = document.getElementById("runBtn");
  const loading = document.getElementById("loadingOverlay");
  runBtn.disabled = true;
  loading.style.display = "flex";

  try {
    const formData = new FormData();
    if (selectedFile) {
      formData.append("file", selectedFile);
    } else {
      formData.append("sample_name", document.getElementById("sampleSelect").value);
    }

    formData.append("model_name", document.getElementById("modelSelect").value);
    formData.append("conf", document.getElementById("confThres").value);
    formData.append("iou", document.getElementById("iouThres").value);
    formData.append("slice_inference", document.getElementById("sliceToggle").checked);
    formData.append("tile_size", document.getElementById("tileSize").value);
    formData.append("overlap", document.getElementById("overlapRatio").value);
    formData.append("target", document.getElementById("targetClassSelect").value);

    const res = await fetch("/api/detect", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!data.success) {
      alert("解译失败: " + (data.error || "未知异常"));
      return;
    }

    // 更新界面渲染
    updateResultsUI(data);
  } catch (err) {
    alert("请求网络异常: " + err.message);
  } finally {
    runBtn.disabled = false;
    loading.style.display = "none";
  }
}

function updateResultsUI(data) {
  // 1. 更新图像
  const imgResult = document.getElementById("imgResult");
  imgResult.src = data.artifacts.annotated_image + "?t=" + new Date().getTime();

  // 2. 更新元数据标签
  const meta = data.image_info;
  const crsStr = meta.crs ? meta.crs : "无地理坐标";
  document.getElementById("imageMetaBar").textContent = 
    `分辨率: ${meta.width} x ${meta.height} | CRS: ${crsStr}`;

  // 3. 更新统计卡片
  const summary = data.summary;
  document.getElementById("totalTargets").textContent = summary.total_objects;
  document.getElementById("inferTime").textContent = (summary.inference_time_ms / 1000).toFixed(1) + "s";

  // 4. 更新图表
  const counts = summary.class_counts || {};
  const labels = Object.keys(counts);
  const values = Object.values(counts);

  if (labels.length > 0) {
    currentChart.data.labels = labels;
    currentChart.data.datasets[0].data = values;
    currentChart.update();
  }

  // 5. 更新明细表格
  const tbody = document.getElementById("targetTableBody");
  tbody.innerHTML = "";
  (data.detections || []).forEach(d => {
    const tr = document.createElement("tr");
    const colorHex = rgbToHex(d.color || [0, 255, 0]);
    const geoStr = d.center_geo ? `${d.center_geo[0]}, ${d.center_geo[1]}` : "-";
    tr.innerHTML = `
      <td>${d.id}</td>
      <td><span class="badge" style="background:${colorHex};">${d.class_name_zh}</span></td>
      <td>${d.confidence.toFixed(2)}</td>
      <td>${d.area_pixels.toLocaleString()}</td>
      <td title="${geoStr}">${geoStr}</td>
    `;
    tbody.appendChild(tr);
  });

  // 6. 更新下载链接
  document.getElementById("btnDlImg").href = data.artifacts.annotated_image;
  document.getElementById("btnDlGeoJson").href = data.artifacts.geojson;
  document.getElementById("btnDlCsv").href = data.artifacts.csv;
  document.getElementById("btnDlJson").href = data.artifacts.json;
  document.getElementById("btnDlReport").href = data.artifacts.report;
}

function rgbToHex(rgb) {
  return "#" + rgb.map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  }).join("");
}
