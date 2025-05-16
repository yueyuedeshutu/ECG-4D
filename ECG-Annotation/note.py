#!/usr/bin/env python3
"""
并行生成心电图 JSON 注释文件脚本

从 machine_measurements.csv、record_list_hr_hrv_test.csv、
record_list_interval_test.csv、record_list_morphology_test.csv、
record_list_structural_test.csv 中读取数据，按照模板生成注释，
使用所有可用 CPU 并行处理，并通过 tqdm 显示进度。
"""

import os
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# —— 全局变量，占位，后面在 __main__ 中赋值 —— #
df_machine = df_hrhrv = df_interval = df_morph = df_struct = None
hr_map = interval_map = morph_map = struct_map = None
output_dir = None

def safe(value):
    """
    将 pandas/NumPy 类型转换为原生 Python 类型，并把 NaN 转为 None，
    以保证 json.dump 能正确输出 null/true/false/int/float。
    """
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value

def process_idx(idx):
    """
    工作进程函数：根据 machine_measurements 中的行号 idx
    构建 JSON 注释并写入文件。
    """
    # 取出该条 machine_measurements 行
    row = df_machine.iloc[idx]
    subj = row['subject_id']
    study = row['study_id']
    num_str = f"{idx:07d}"

    # 从 map 中查找其它表对应记录
    hr_row       = hr_map.get((subj, study))
    interval_row = interval_map.get((subj, study))
    morph_row    = morph_map.get((subj, study))
    struct_row   = struct_map.get((subj, study))

    # 构建 JSON 结构
    data = {
        "number": num_str,
        "patient_info": {
            "demographics": {
                "gender":   None,
                "age":      None,
                "height_cm":None,
                "weight_kg":None,
                "bmi":      None
            },
            "clinical_history": {
                "comorbidities":   [],
                "cardiac_history": [],
                "medications":     []
            }
        },
        "metadata": {
            "data_source": {
                "original_dataset": "MIMIC-IV-ECG",
                "record_id":        safe(subj)
            },
            "lead": {
                "num_leads": 12,
                "lead_names": ["I","II","III","aVR","aVF","aVL","V1","V2","V3","V4","V5","V6"]
            },
            "sampling_rate":   500,
            "signal_duration": 5000
        },
        "expert_annotations": {
            "source_file":      None,
            "lead_annotations": None
        },
        "diagnostic_metrics": {
            "rhythm_metrics": {
                "heart_rate": {
                    "mean_HR": safe(hr_row['HR']) if hr_row else None,
                    "HRV": {
                        "time_domain": {
                            "SDNN":  safe(hr_row['SDNN']) if hr_row else None,
                            "RMSSD": safe(hr_row['RMSSD']) if hr_row else None,
                            "pNN50": safe(hr_row['pNN50']) if hr_row else None,
                        },
                        "frequency_domain": {
                            "LF":          safe(hr_row['LF'])    if hr_row else None,
                            "HF":          safe(hr_row['HF'])    if hr_row else None,
                            "LF_HF_ratio": safe(hr_row['LF_HF']) if hr_row else None,
                        }
                    }
                }
            },
            "interval_metrics": {
                "RR_interval": {
                    "mean":  safe(interval_row['RR_mean'])   if interval_row else None,
                    "SDNN":  safe(interval_row['RR_SDNN'])   if interval_row else None,
                    "RMSSD": safe(interval_row['RR_RMSSD'])  if interval_row else None,
                },
                "PR_interval":   safe(interval_row['PR_interval'])   if interval_row else None,
                "QRS_duration":  safe(interval_row['QRS_duration'])  if interval_row else None,
                "QT_interval":   safe(interval_row['QT_interval'])   if interval_row else None,
                "QTc_interval":  safe(interval_row['QTc_interval'])  if interval_row else None,
                "QT_dispersion": safe(interval_row['QT_dispersion']) if interval_row else None,
                "JT_interval":   safe(interval_row['JT_interval'])   if interval_row else None,
                "Tpe_interval":  safe(interval_row['Tpe_interval'])  if interval_row else None,
            },
            "morphology_metrics": {
                "P_wave": {
                    "amplitude": safe(morph_row['P_amp_mean'])    if morph_row else None,
                    "duration":  safe(morph_row['P_dur_mean']*1000) if morph_row else None,
                    "dispersion":safe(morph_row['P_dispersion']*1000) if morph_row else None,
                    "notching":  safe(morph_row['P_notch'])       if morph_row else None,
                },
                "QRS_complex": {
                    "duration":            safe(int(struct_row['qrs_duration_ms'])) if struct_row else None,
                    "R_amplitude":         safe(morph_row['R_amp_mean'])            if morph_row else None,
                    "S_depth":             safe(morph_row['S_depth_mean'])          if morph_row else None,
                    "fragmented_qrs":      safe(morph_row['fQRS'])                  if morph_row else None,
                    "pathological_q_wave": safe(morph_row['Pathological_Q'])        if morph_row else None,
                    "R/S_ratio": {
                        "R/S_V1": safe(morph_row['RS_V1']) if morph_row else None,
                        "R/S_V2": safe(morph_row['RS_V2']) if morph_row else None,
                        "R/S_V3": safe(morph_row['RS_V3']) if morph_row else None,
                        "R/S_V4": safe(morph_row['RS_V4']) if morph_row else None,
                        "R/S_V5": safe(morph_row['RS_V5']) if morph_row else None,
                        "R/S_V6": safe(morph_row['RS_V6']) if morph_row else None,
                    }
                },
                "ST_T_wave": {
                    "ST_deviation":    safe(morph_row['ST_dev_mean'])  if morph_row else None,
                    "ST_segment_slope":safe(morph_row['ST_slope_mean'])if morph_row else None,
                    "T_amplitude":     safe(morph_row['T_amp_mean'])    if morph_row else None,
                    "T_symmetry":      safe(morph_row['T_sym_mean'])    if morph_row else None,
                },
                "U_wave": {
                    "amplitude": safe(morph_row['U_amp_mean']) if morph_row else None
                }
            },
            "vector_metrics": {
                "frontal_axis": {
                    "QRS_axis":    safe(interval_row['QRS_axis'])    if interval_row else None,
                    "T_axis":      safe(interval_row['T_axis'])      if interval_row else None,
                    "QRS_T_angle": safe(interval_row['QRS_T_angle'])if interval_row else None,
                },
                "horizontal_axis": {
                    "RS_transition_lead": safe(interval_row['RS_transition_lead']) if interval_row else None,
                    "R_wave_slope":       safe(interval_row['R_wave_slope'])     if interval_row else None,
                }
            },
            "structural_metrics": {
                "Sokolow_Lyon":   safe(struct_row['Sokolow_Lyon_Index'])   if struct_row else None,
                "Cornell_Voltage":safe(struct_row['Cornell_Voltage_Index'])if struct_row else None,
                "Cornell_Product":safe(struct_row['Cornell_Product_Index'])if struct_row else None,
            }
        },
        "signal_quality": {
            "Baseline_Wander":       safe(struct_row['Baseline_Wander'])      if struct_row else None,
            "Static_Noise":          safe(struct_row['Static_Noise'])         if struct_row else None,
            "Burst_Noise":           safe(struct_row['Burst_Noise'])          if struct_row else None,
            "Electrode_Issue":       safe(struct_row['Electrode_Issue'])      if struct_row else None,
            "Powerline_Interference":safe(struct_row['Powerline_Interference'])if struct_row else None,
            "Signal_Quality":        safe(struct_row['Signal_Quality'])       if struct_row else None,
        },
        "diagnostic_conclusion": {
            "primary_diagnosis":        safe(row.get('report_0')),
            "secondary_diagnosis":      safe(row.get('report_1')),
            "supplementary_description":safe(row.get('report_2')),
            "extended_annotations": [
                safe(row[f'report_{i}'])
                for i in range(3, 18)
                if pd.notna(row.get(f'report_{i}', np.nan)) and row.get(f'report_{i}') != ""
            ]
        },
        "data_paths": {
            "raw_data":       safe(struct_row['path']) if struct_row else None,
            "annotation_file":None,
            "ecg_plot":       f"{num_str}.png"
        }
    }

    # 写入 JSON 文件
    out_path = os.path.join(output_dir, f"{num_str}.json")
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

    return None  # 只为 tqdm 进度计数使用

if __name__ == '__main__':
    # —— 加载 CSV 数据 —— #
    machine_csv    = './machine_measurements.csv'
    hrhrv_csv      = './record_list_hr_hrv.csv'
    interval_csv   = './record_list_interval.csv'
    morphology_csv = './record_list_morphology.csv'
    structural_csv = './record_list_structural.csv'
    output_dir     = './ECG/note'
    os.makedirs(output_dir, exist_ok=True)

    df_machine  = pd.read_csv(machine_csv,    parse_dates=['ecg_time'], low_memory=False)
    df_hrhrv    = pd.read_csv(hrhrv_csv,      parse_dates=['ecg_time'], low_memory=False)
    df_interval = pd.read_csv(interval_csv,   parse_dates=['ecg_time'], low_memory=False)
    df_morph    = pd.read_csv(morphology_csv, low_memory=False)
    df_struct   = pd.read_csv(structural_csv, parse_dates=['ecg_time'], low_memory=False)

    # —— 构建快速查找字典 —— #
    df_hrhrv    .set_index(['subject_id','study_id'], inplace=True)
    df_interval .set_index(['subject_id','study_id'], inplace=True)
    df_morph    .set_index(['subject_id','study_id'], inplace=True)
    df_struct   .set_index(['subject_id','study_id'], inplace=True)

    hr_map       = df_hrhrv   .to_dict('index')
    interval_map = df_interval.to_dict('index')
    morph_map    = df_morph    .to_dict('index')
    struct_map   = df_struct   .to_dict('index')

    total = len(df_machine)
    print(f"开始生成 {total} 条 JSON 注释，使用 {cpu_count()} 个进程...")

    # —— 并行处理 —— #
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap(process_idx, range(total)), total=total):
            pass

    print("完成：所有 JSON 注释已生成至", output_dir)
