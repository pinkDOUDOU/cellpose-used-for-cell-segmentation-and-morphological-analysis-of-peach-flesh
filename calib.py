import pandas as pd
import numpy as np
from skimage import io, measure
from cellpose import models, core, io as cpio
from pathlib import Path

# ---------- 1. 标定 ----------
cpio.logger_setup()
if not core.use_gpu():
    raise RuntimeError("无 GPU，请检查环境")

cal_path = Path("calibration.jpg")  # 带 200 µm 标尺的图
if not cal_path.exists():
    raise FileNotFoundError("请把 calibration.jpg 放在当前目录")

model = models.CellposeModel(gpu=True)
diameter = 60  # 与分割时一致
cal_img = io.imread(cal_path)

masks, *_ = model.eval(
    cal_img,
    diameter=diameter,
    flow_threshold=0.35,
    cellprob_threshold=0.0,
    normalize={"tile_norm_blocksize": 0}
)

props = measure.regionprops_table(masks, properties=['feret_diameter_max'])
L_px = np.max(props['feret_diameter_max'])
k = 200.0 / L_px  # µm/pixel
print(f"标定完成：k = {k:.4f} µm/pixel")

# ---------- 2. 定义需要换算的列 ----------
length_cols = [
    'equivalent_diameter_area',
    'perimeter',
    'perimeter_crofton',
    'axis_major_length',
    'axis_minor_length',
    'feret_diameter_max'
]

area_cols = [
    'area',
    'mean_area',
    'ics_area',
    'num_pixels'
]

# ---------- 3. 遍历子文件夹 ----------
root = Path("./")  # 根目录（包含 NB-D1, NB-D2, ... NB-P5 的目录）

for subdir in root.iterdir():
    if subdir.is_dir() and subdir.name.startswith(("NB-D", "NB-P")):
        old_csv = subdir / "segmentation_summary.csv"
        if not old_csv.exists():
            print(f"⚠️ 跳过 {subdir} (没有 segmentation_summary.csv)")
            continue

        print(f"\n处理文件夹: {subdir.name}")
        df = pd.read_csv(old_csv)

        # 长度类（像素 → µm）
        for col in length_cols:
            if col in df.columns:
                df[col + '_um'] = df[col] * k

        # 面积类（像素² → µm²）
        for col in area_cols:
            if col in df.columns:
                df[col + '_um2'] = df[col] * (k ** 2)

        # 保存结果
        new_csv = subdir / "segmentation_summary_real_unit.csv"
        df.to_csv(new_csv, index=False)
        print(f"  → 已保存: {new_csv}")
