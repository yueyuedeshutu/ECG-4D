# mimic_iv_ecg_hrv.py
# -*- coding: utf-8 -*-
"""
批量读取 MIMIC-IV-ECG 数据集，计算心率 (HR) 及心率变异性 (HRV) 指标：
SDNN、RMSSD、pNN50、LF、HF、LF/HF Ratio。
"""
import os
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.signal import welch
from scipy.interpolate import interp1d
from tqdm import tqdm


def peaks_to_rr(peaks_idx, fs):
    """
    从 R 峰索引和采样率计算 RR 间期（ms）及对应的时间点（s）。
    peaks_idx: 1D array of R 峰样本点索引
    fs: 采样率 (Hz)
    返回 rr_ms, rr_times
    """
    times_s = np.asarray(peaks_idx) / fs
    rr_ms = np.diff(times_s) * 1000
    rr_times = times_s[1:]
    return rr_ms, rr_times


def compute_time_domain(rr_ms):
    """
    计算时域 HRV 指标：SDNN, RMSSD, pNN50
    rr_ms: R-R 间期序列 (ms)
    返回 dict
    """
    rr = rr_ms[~np.isnan(rr_ms)]
    N = len(rr)
    if N < 2:
        return {'SDNN': np.nan, 'RMSSD': np.nan, 'pNN50': np.nan}
    mean_rr = np.mean(rr)
    sdnn = np.sqrt(np.sum((rr - mean_rr)**2) / (N - 1))
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.sum(diff_rr**2) / (N - 1))
    pnn50 = 100.0 * np.sum(np.abs(diff_rr) > 50) / (N - 1)
    return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50}


def compute_frequency_domain(rr_ms, rr_times, fs_interp=4.0):
    """
    计算频域 HRV 指标：LF, HF, LF/HF Ratio
    rr_ms: R-R 间期序列 (ms)
    rr_times: 对应时间点 (s)
    fs_interp: 插值采样率 (Hz)
    返回 dict
    """
    # 数据量过少时返回 NaN
    if len(rr_ms) < 2:
        return {'LF': np.nan, 'HF': np.nan, 'LF/HF': np.nan}

    # 插值到等间距时间轴
    try:
        interp_func = interp1d(rr_times, rr_ms, kind='linear', fill_value='extrapolate')
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0/fs_interp)
        rr_uniform = interp_func(t_uniform)
    except Exception:
        return {'LF': np.nan, 'HF': np.nan, 'LF/HF': np.nan}

    # 去趋势
    rr_detrended = rr_uniform - np.mean(rr_uniform)

    # Welch 估计功率谱
    f, pxx = welch(rr_detrended, fs=fs_interp, nperseg=min(256, len(rr_detrended)))
    if len(f) < 2:
        return {'LF': np.nan, 'HF': np.nan, 'LF/HF': np.nan}

    # 频段功率近似：带宽 * 单位功率和
    df = f[1] - f[0]
    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f >= 0.15) & (f <= 0.40)
    lf = np.sum(pxx[lf_mask]) * df
    hf = np.sum(pxx[hf_mask]) * df
    lf_hf = lf / hf if hf > 0 else np.nan
    return {'LF': lf, 'HF': hf, 'LF/HF': lf_hf}


def analyse_record(record_path, sampling_rate=500):
    """
    读取单条心电图记录，计算 HR 与 HRV 指标。
    record_path: 不含扩展名的 wfdb 记录路径
    返回指标元组: HR, SDNN, RMSSD, pNN50, LF, HF, LF/HF
    """
    try:
        record = wfdb.rdsamp(record_path)[0]
        signal = record.T[1]
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        peaks = rpeaks.get('ECG_R_Peaks', [])
        rr_ms, rr_times = peaks_to_rr(peaks, sampling_rate)
        hr = 60.0 / (np.mean(rr_ms) / 1000.0) if len(rr_ms) > 0 else np.nan
        td = compute_time_domain(rr_ms)
        fd = compute_frequency_domain(rr_ms, rr_times)
        return hr, td['SDNN'], td['RMSSD'], td['pNN50'], fd['LF'], fd['HF'], fd['LF/HF']
    except Exception:
        return [np.nan] * 7


def main(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'record_list.csv'))
    cols = ['HR', 'SDNN', 'RMSSD', 'pNN50', 'LF', 'HF', 'LF_HF']
    for col in cols:
        df[col] = np.nan

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rec_path = os.path.join(data_dir, row['path'])
        metrics = analyse_record(rec_path, sampling_rate=500)
        df.loc[idx, cols] = metrics

    out_file = os.path.join(data_dir, 'record_list_hr_hrv.csv')
    df.to_csv(out_file, index=False)
    print(f"已保存: {out_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='批量计算 MIMIC-IV-ECG HR & HRV 指标')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='MIMIC-IV-ECG 数据集根目录')
    args = parser.parse_args()
    main(args.data_dir)
