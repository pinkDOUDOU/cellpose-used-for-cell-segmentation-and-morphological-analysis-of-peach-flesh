import numpy as np
from cellpose import models, core, io, plot
from pathlib import Path
from natsort import natsorted
import imageio
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd

# ---------- 基础配置 ----------
io.logger_setup()
if not core.use_gpu():
    raise ImportError("无 GPU，请检查运行环境")

model = models.CellposeModel(gpu=True)

dir_path = Path(r"C:/Users/YUAN/Desktop/linshi-NB-seg-cell/NB-P4")  # ← 改成你的路径
image_ext = ".jpg"
flow_threshold = 0.35
cellprob_threshold = 0.0
diameter = 110

figures_dir = dir_path / "seg-resultes"
figures_dir.mkdir(exist_ok=True)

files = natsorted([f for f in dir_path.glob(f"*{image_ext}")
                   if "_masks" not in f.name and "_flows" not in f.name])

summary_data = []

# ---------- 批量处理 ----------
for f in files:
    print(f"Processing {f.name}...")
    img = io.imread(f)

    masks, flows, styles = model.eval(
        img,
        batch_size=32,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": 0}
    )

    # 保存 mask / 彩色 mask / flow / 原图
    io.imsave(figures_dir / f"{f.stem}_masks.tif", masks.astype("uint16"))
    imageio.imwrite(figures_dir / f"{f.stem}_color_mask.png", plot.mask_rgb(masks))
    imageio.imwrite(figures_dir / f"{f.stem}_flow.png", flows[0])
    imageio.imwrite(figures_dir / f"{f.stem}_original.png", img)

    # ---------- 细胞指标 ----------
    props = measure.regionprops_table(
        masks,
        properties=['area', 'equivalent_diameter_area', 'perimeter', 'perimeter_crofton',
                    'axis_major_length', 'axis_minor_length', 'solidity',
                    'eccentricity', 'extent', 'feret_diameter_max', 'num_pixels']
    )
    df = pd.DataFrame(props)

    total_cell_area = df['area'].sum()
    total_img_area = img.shape[0] * img.shape[1]
    ics_area = total_img_area - total_cell_area
    ics_ratio = ics_area / total_img_area

    circularity = (4 * np.pi * df['area']) / (df['perimeter_crofton'] ** 2)
    Rs = total_cell_area / total_img_area  # 孔隙率

    summary_data.append({
        "filename": f.name,
        "cell_count": len(df),
        "mean_area": df['area'].mean(),
        "equivalent_diameter_area": df['equivalent_diameter_area'].mean(),
        "perimeter": df['perimeter'].mean(),
        "perimeter_crofton": df['perimeter_crofton'].mean(),
        "axis_major_length": df['axis_major_length'].mean(),
        "axis_minor_length": df['axis_minor_length'].mean(),
        "solidity": df['solidity'].mean(),
        "eccentricity": df['eccentricity'].mean(),
        "extent": df['extent'].mean(),
        "feret_diameter_max": df['feret_diameter_max'].mean(),
        "num_pixels": df['num_pixels'].mean(),
        "roundness(circularity)": circularity.mean(),
        "porosity(Rs)": Rs,
        "ics_area": ics_area,
        "ics_ratio": ics_ratio
    })

# ---------- 保存汇总 ----------
summary_df = pd.DataFrame(summary_data)
summary_csv = dir_path / "segmentation_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print("All done! Summary saved to:", summary_csv)