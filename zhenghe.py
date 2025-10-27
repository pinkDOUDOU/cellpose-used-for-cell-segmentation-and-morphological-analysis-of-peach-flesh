import os
import pandas as pd
import re

# ---------- 工具函数 ----------
def extract_info_from_filename(filename):
    pattern = r'NB-([DP]\d+)-(\d+).*?-(\d+)\.jpg'
    match = re.search(pattern, filename)
    if match:
        day    = 'NB-' + match.group(1)
        sample = match.group(1) + '-' + match.group(2).zfill(2)
        region = int(match.group(3))
        return day, sample, region
    return None, None, None

def process_folder(folder_path):
    csv_path = os.path.join(folder_path, 'segmentation_summary_real_unit.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    folder_name = os.path.basename(folder_path)
    df['Day'] = folder_name
    df['Sample'] = ''
    df['Region'] = 0
    first_col = df.columns[0]
    for idx, filename in enumerate(df[first_col]):
        day, sample, region = extract_info_from_filename(filename)
        if day:
            df.at[idx, 'Sample'] = sample
            df.at[idx, 'Region'] = region
    return df

# ---------- 主流程 ----------
def main():
    root_folder = '.'          # 当前文件夹
    all_data = []
    for subfolder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(folder_path):
            print(f'Processing {subfolder}...')
            df = process_folder(folder_path)
            if df is not None:
                all_data.append(df)

    if not all_data:
        print('No valid data found!')
        return

    # 合并并计算“样品”级平均
    combined = pd.concat(all_data, ignore_index=True)
    param_cols = [c for c in combined.columns
                  if c not in {combined.columns[0], 'Day', 'Sample', 'Region'}]
    sample_avg = (combined.groupby(['Day', 'Sample'])[param_cols]
                          .mean()
                          .reset_index())

    # 保存样品级结果
    sample_avg.to_csv('integrated_morphology_data.csv', index=False)
    print('Saved: integrated_morphology_data.csv')

    # 计算“天数”级均值与标准误（分开列）
    day_records = []
    for day, grp in sample_avg.groupby('Day'):
        record = {'Day': day}
        for p in param_cols:
            record[f'{p}_mean'] = grp[p].mean()
            record[f'{p}_sem']  = grp[p].sem()
        day_records.append(record)

    daily_df = pd.DataFrame(day_records)
    daily_df.to_csv('daily_mean_sem.csv', index=False)
    print('Saved: daily_mean_sem.csv (mean & sem in separate columns)')

if __name__ == '__main__':
    main()