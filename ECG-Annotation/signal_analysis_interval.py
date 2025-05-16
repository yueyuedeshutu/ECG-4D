#!/usr/bin/env python3
# signal_analysis_interval.py

import os
import warnings
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from tqdm import tqdm
from scipy.stats import linregress

# 屏蔽 neurokit2 的所有 warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 对单导联信号计算 QRS 与 T 波幅值
def compute_qrs_amplitude(lead_signal, rpeaks, fs, window_ms=100):
    window_samples = int(window_ms * fs / 1000)
    amplitudes = []
    for r in rpeaks:
        try:
            r = int(r)
            start = max(0, r - window_samples)
            end = min(len(lead_signal), r + window_samples)
            seg = lead_signal[start:end]
            if len(seg) > 0:
                amplitude = np.max(seg) - np.min(seg)
                if not np.isnan(amplitude):
                    amplitudes.append(amplitude)
        except Exception:
            continue
    return np.mean(amplitudes) if amplitudes else np.nan

def compute_t_amplitude(lead_signal, rpeaks, fs, offset_ms=300, window_ms=100):
    window_samples = int(window_ms * fs / 1000)
    offset_samples = int(offset_ms * fs / 1000)
    amplitudes = []
    for r in rpeaks:
        try:
            r = int(r)
            start = r + offset_samples
            end = min(len(lead_signal), start + window_samples)
            seg = lead_signal[start:end]
            if len(seg) > 0:
                amplitude = np.max(seg) - np.min(seg)
                if not np.isnan(amplitude):
                    amplitudes.append(amplitude)
        except Exception:
            continue
    return np.mean(amplitudes) if amplitudes else np.nan

def compute_qt_dispersion(full_signal, fs, sig_names, common_rpeaks):
    qt_vals = []
    for i, lead_name in enumerate(sig_names):
        try:
            lead_signal = full_signal[:, i]
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)
            _, waves = nk.ecg_delineate(cleaned, common_rpeaks, sampling_rate=fs, method="dwt", show=False)
            qt_intervals = []
            for j in range(len(common_rpeaks.get("ECG_R_Peaks", []))):
                if j < len(waves.get("ECG_T_Offsets", [])) and j < len(waves.get("ECG_R_Onsets", [])):
                    qt = (waves["ECG_T_Offsets"][j] - waves["ECG_R_Onsets"][j]) / fs
                    if not np.isnan(qt):
                        qt_intervals.append(qt)
            if qt_intervals:
                median_qt = np.median(qt_intervals)
                qt_vals.append(median_qt)
        except Exception:
            continue
    if qt_vals:
        return (max(qt_vals) - min(qt_vals)) * 1000
    return np.nan

def read_record(record_path):
    try:
        record = wfdb.rdrecord(record_path)
        return record.p_signal, record.fs, record.sig_name
    except Exception as e:
        print(f"读取记录 {record_path} 失败：{e}")
        return None, None, None

def process_lead(signal, sampling_rate=500):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
    _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt", show=False)
    return ecg_cleaned, rpeaks, waves

def compute_time_intervals(rpeaks, waves, sampling_rate=500):
    rr_intervals, pr_intervals, qrs_durations = [], [], []
    qt_intervals, jt_intervals, tpe_intervals = [], [], []
    r_peaks = rpeaks.get("ECG_R_Peaks", [])

    for i in range(len(r_peaks)):
        if i > 0:
            rr = (r_peaks[i] - r_peaks[i-1]) / sampling_rate
            if rr > 0:
                rr_intervals.append(rr)
        try:
            pr = (waves["ECG_Q_Peaks"][i] - waves["ECG_P_Onsets"][i]) / sampling_rate
            pr_intervals.append(pr)
        except: pass
        try:
            qrs = (waves["ECG_S_Peaks"][i] - waves["ECG_Q_Peaks"][i]) / sampling_rate
            qrs_durations.append(qrs)
        except: pass
        try:
            qt = (waves["ECG_T_Offsets"][i] - waves["ECG_R_Onsets"][i]) / sampling_rate
            qt_intervals.append(qt)
        except: pass
        if qt_intervals and qrs_durations:
            jt = qt_intervals[-1] - qrs_durations[-1]
            jt_intervals.append(jt)
        try:
            tpe = (waves["ECG_T_Offsets"][i] - waves["ECG_T_Peaks"][i]) / sampling_rate
            tpe_intervals.append(tpe)
        except: pass

    def safe_mean(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return np.mean(arr) if arr.size > 0 else np.nan

    avg_rr = safe_mean(rr_intervals)
    avg_qt = safe_mean(qt_intervals)
    qtc = avg_qt / np.sqrt(avg_rr) if avg_rr and avg_rr > 0 else np.nan

    return {
        "RR_mean": avg_rr * 1000 if avg_rr else np.nan,
        "RR_SDNN": np.std(rr_intervals) * 1000 if len(rr_intervals) > 1 else np.nan,
        "RR_RMSSD": np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000 if len(rr_intervals) > 1 else np.nan,
        "PR_interval": safe_mean(pr_intervals) * 1000,
        "QRS_duration": safe_mean(qrs_durations) * 1000,
        "QT_interval": avg_qt * 1000,
        "QTc_interval": qtc * 1000 if not np.isnan(qtc) else np.nan,
        "QT_dispersion": np.nan,
        "JT_interval": safe_mean(jt_intervals) * 1000,
        "Tpe_interval": safe_mean(tpe_intervals) * 1000
    }

def compute_hr_metrics(rpeaks, sampling_rate=500):
    rpeaks_indices = np.array(rpeaks.get("ECG_R_Peaks", []))
    if len(rpeaks_indices) < 2:
        return {"HR": np.nan, "SDNN": np.nan, "RMSSD": np.nan}
    rr = np.diff(rpeaks_indices) / sampling_rate
    rr = rr[rr > 0]
    return {
        "HR": 60 / np.mean(rr) if len(rr) > 0 else np.nan,
        "SDNN": np.std(rr) * 1000,
        "RMSSD": np.sqrt(np.mean(np.diff(rr)**2)) * 1000
    }

def compute_spatial_metrics(full_signal, fs, sig_names, common_rpeaks):
    spatial = {"QRS_axis": np.nan, "T_axis": np.nan, "QRS_T_angle": np.nan, "RS_transition_lead": np.nan, "R_wave_slope": np.nan}
    try:
        idx_I = sig_names.index("I")
        idx_aVF = sig_names.index("aVF")
        lead_I = full_signal[:, idx_I]
        lead_aVF = full_signal[:, idx_aVF]
        avg_qrs_I = compute_qrs_amplitude(lead_I, common_rpeaks["ECG_R_Peaks"], fs)
        avg_qrs_aVF = compute_qrs_amplitude(lead_aVF, common_rpeaks["ECG_R_Peaks"], fs)
        qrs_axis = np.degrees(np.arctan2(avg_qrs_aVF, avg_qrs_I))
        spatial["QRS_axis"] = (qrs_axis + 180) % 360 - 180

        avg_t_I = compute_t_amplitude(lead_I, common_rpeaks["ECG_R_Peaks"], fs)
        avg_t_aVF = compute_t_amplitude(lead_aVF, common_rpeaks["ECG_R_Peaks"], fs)
        t_axis = np.degrees(np.arctan2(avg_t_aVF, avg_t_I))
        spatial["T_axis"] = (t_axis + 180) % 360 - 180

        spatial["QRS_T_angle"] = abs(spatial["QRS_axis"] - spatial["T_axis"])
    except: pass

    V_leads = []
    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        try:
            V_leads.append(full_signal[:, sig_names.index(lead)])
        except: continue

    if len(V_leads) == 6:
        R_amp = [np.percentile(lead, 95) for lead in V_leads]
        S_amp = [np.percentile(lead, 5) for lead in V_leads]
        for i in range(6):
            if R_amp[i] > abs(S_amp[i]):
                spatial["RS_transition_lead"] = f"V{i+1}"
                break
        slope, _, _, _, _ = linregress(np.arange(1, 7), R_amp)
        spatial["R_wave_slope"] = slope

    return spatial

def process_record(record_path, sampling_rate=500):
    full_signal, fs, sig_names = read_record(record_path)
    if full_signal is None:
        return None
    fs = fs or sampling_rate
    try:
        idx_leadII = sig_names.index("II") if "II" in sig_names else 0
    except:
        idx_leadII = 0
    signal_leadII = full_signal[:, idx_leadII]
    try:
        ecg_cleaned, rpeaks, waves = process_lead(signal_leadII, fs)
    except Exception as e:
        print(f"处理 {record_path} 出错: {e}")
        return None

    time_intervals = compute_time_intervals(rpeaks, waves, fs)
    time_intervals["QT_dispersion"] = compute_qt_dispersion(full_signal, fs, sig_names, rpeaks)
    hr_metrics = compute_hr_metrics(rpeaks, fs)
    spatial_metrics = compute_spatial_metrics(full_signal, fs, sig_names, rpeaks)

    all_metrics = {**time_intervals, **hr_metrics, **spatial_metrics}
    return all_metrics
