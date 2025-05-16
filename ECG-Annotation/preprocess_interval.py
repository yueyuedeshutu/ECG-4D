#!/usr/bin/env python3
import os
import pandas as pd
from tqdm import tqdm
import argparse
import multiprocessing
from functools import partial
from signal_analysis_interval import process_record

def process_and_update(row, data_dir):
    rec_rel_path = row["path"]
    record_path = os.path.join(data_dir, rec_rel_path)
    try:
        metrics = process_record(record_path, sampling_rate=500)
    except Exception as e:
        print(f"处理 {record_path} 出错：{e}")
        metrics = None

    result = row.to_dict()
    if metrics:
        result.update(metrics)
    else:
        # 出错时所有新指标设为 None
        for metric in [
            "HR", "SDNN", "RMSSD", "RR_mean", "RR_SDNN", "RR_RMSSD",
            "PR_interval", "QRS_duration", "QT_interval", "QTc_interval",
            "QT_dispersion", "JT_interval", "Tpe_interval",
            "QRS_axis", "T_axis", "QRS_T_angle", "RS_transition_lead", "R_wave_slope"
        ]:
            result[metric] = None
    return result

def prepare(args):
    data_dir = args.data_dir
    record_list_path = os.path.join(data_dir, "record_list_test.csv")
    df = pd.read_csv(record_list_path)

    # 添加新指标列
    new_metrics = ["HR", "SDNN", "RMSSD", "RR_mean", "RR_SDNN", "RR_RMSSD",
                   "PR_interval", "QRS_duration", "QT_interval", "QTc_interval",
                   "QT_dispersion", "JT_interval", "Tpe_interval",
                   "QRS_axis", "T_axis", "QRS_T_angle", "RS_transition_lead", "R_wave_slope"]
    for metric in new_metrics:
        if metric not in df.columns:
            df[metric] = None

    print("开始并行处理 ECG 记录...")
    num_cpus = multiprocessing.cpu_count()
    print(f"检测到 CPU 核心数：{num_cpus}，使用全部核心进行处理。")

    with multiprocessing.Pool(processes=num_cpus) as pool:
        worker = partial(process_and_update, data_dir=data_dir)
        results = list(tqdm(pool.imap(worker, [row for _, row in df.iterrows()]), total=len(df)))

    new_df = pd.DataFrame(results)
    output_path = os.path.join(data_dir, "record_list_interval_test.csv")
    new_df.to_csv(output_path, index=False)
    print(f"处理完成，结果保存至：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并行预处理 MIMIC-IV-ECG 数据集，提取心电指标。")
    parser.add_argument("--data-dir", type=str, default="", help="数据集根目录路径")
    args = parser.parse_args()
    prepare(args)
