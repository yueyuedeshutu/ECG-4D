#!/usr/bin/env python3
# morphology_features.py

import os
import warnings
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# Suppress unwanted warnings
warnings.filterwarnings("ignore", message="Too few peaks detected to compute the rate.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Safe statistics
safe_mean = lambda x: np.mean(x) if x.size > 0 else np.nan
safe_diff = lambda x: (np.max(x) - np.min(x)) if x.size > 0 else np.nan

class MIMICIVECGLoader:
    """
    Data loader for the MIMIC-IV-ECG diagnostic subset.
    """
    def __init__(self, root_dir: str):
        self.root = root_dir
        self.record_list = pd.read_csv(os.path.join(self.root, 'record_list.csv'))

    def get_waveform_path(self, subject_id: int, study_id: int) -> str:
        rec = self.record_list[
            (self.record_list.subject_id == subject_id) &
            (self.record_list.study_id == study_id)
        ]
        if rec.empty:
            raise ValueError(f"No record for {subject_id}-{study_id}")
        return os.path.join(self.root, rec.iloc[0]['path'])

    def load_signal(self, subject_id: int, study_id: int):
        wf = self.get_waveform_path(subject_id, study_id)
        record = wfdb.rdrecord(wf)
        sig = np.nan_to_num(record.p_signal)
        return sig, record


def compute_features_for_entry(entry):
    subject_id, study_id, path, data_dir = entry
    try:
        loader = MIMICIVECGLoader(data_dir)
        sig, record = loader.load_signal(subject_id, study_id)
        lead_names = record.sig_name
        # Reference lead II
        ii_idx = lead_names.index('II')
        ecg2 = sig[:, ii_idx]
        ecg_clean = nk.ecg_clean(ecg2, sampling_rate=500)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=500)
        _, waves = nk.ecg_delineate(ecg_clean, rpeaks,
                                     sampling_rate=500,
                                     method='dwt', show=False, show_type='all')
        R_all = np.array(rpeaks['ECG_R_Peaks'], dtype=int)
        nbeats = len(R_all)
        wave_keys = [k for k, v in waves.items() if v is not None and k != 'ECG_R_Peaks']
        valid_idxs = [i for i in range(nbeats) if all(
            waves[k][i] is not None and not (isinstance(waves[k][i], float) and np.isnan(waves[k][i]))
            for k in wave_keys
        )]
        if not valid_idxs:
            return None
        def wave_arr(key):
            return np.array([waves[key][i] for i in valid_idxs], dtype=int)
        # Extract indices
        Pon, Ppk, Poff = wave_arr('ECG_P_Onsets'), wave_arr('ECG_P_Peaks'), wave_arr('ECG_P_Offsets')
        Qpk, Spk = wave_arr('ECG_Q_Peaks'), wave_arr('ECG_S_Peaks')
        Ron = wave_arr('ECG_R_Onsets') if 'ECG_R_Onsets' in waves else Qpk
        Roff = wave_arr('ECG_R_Offsets') if 'ECG_R_Offsets' in waves else Qpk + int(0.04 * 500)
        Ton, Tpk, Toff = wave_arr('ECG_T_Onsets'), wave_arr('ECG_T_Peaks'), wave_arr('ECG_T_Offsets')
        Rpk = np.array([R_all[i] for i in valid_idxs], dtype=int)

        feats = {}
        # P-wave
        p_amps = ecg_clean[Ppk]
        pdurs = (Poff - Pon) / 500
        feats['P_amp_mean']   = safe_mean(p_amps)
        feats['P_dur_mean']   = safe_mean(pdurs)
        feats['P_dispersion'] = safe_diff(pdurs)
        feats['P_notch']      = bool(any(
            len(find_peaks(ecg_clean[o:f], distance=0.02*500)[0]) > 1
            for o, f in zip(Pon, Poff)
        ))
        # QRS
        r_amps, s_amps = ecg_clean[Rpk], ecg_clean[Spk]
        feats['R_amp_mean']   = safe_mean(r_amps)
        feats['S_depth_mean'] = safe_mean(-s_amps)
        feats['S_amp_mean']   = safe_mean(np.abs(s_amps))
        # R/S per V1-V6
        for v in ['V1','V2','V3','V4','V5','V6']:
            idx = lead_names.index(v)
            clean_v = nk.ecg_clean(sig[:, idx], sampling_rate=500)
            r_vals = clean_v[Rpk]
            s_vals = np.abs(clean_v[Spk])
            mask = s_vals != 0
            feats[f'RS_{v}'] = safe_mean(r_vals[mask] / s_vals[mask]) if mask.any() else np.nan
        # Pathological Q & fQRS
        q_amps = ecg_clean[Qpk]
        q_widths = (Ron - Qpk) / 500
        feats['Pathological_Q'] = bool(np.any(
            (q_widths >= 0.04) | (np.abs(q_amps) >= 0.25 * safe_mean(r_amps))
        ))
        feats['fQRS'] = bool(any(
            len(find_peaks(ecg_clean[on:off], height=0.5*np.max(ecg_clean[on:off]))[0]) > 1
            for on, off in zip(Ron, Roff)
        ))
        # ST-T
        STdev, STslope, Tamp, Tsym, Uamp = [], [], [], [], []
        for i in range(len(Rpk)-1):
            baseline = safe_mean(ecg_clean[Poff[i]:Qpk[i]])
            j = Spk[i]
            STdev.append(ecg_clean[j] - baseline)
            j60 = j + int(0.06 * 500)
            if j60 < len(ecg_clean): STslope.append((ecg_clean[j60] - ecg_clean[j]) / 0.06)
            Tamp.append(ecg_clean[Tpk[i]] - baseline)
            upA = np.trapezoid(ecg_clean[Ton[i]:Tpk[i]], dx=1/500)
            dnA = np.trapezoid(ecg_clean[Tpk[i]:Toff[i]], dx=1/500)
            Tsym.append(upA / dnA if dnA else np.nan)
            seg_u = ecg_clean[Ton[i] + int(0.05*500): min(len(ecg_clean), Toff[i] + int(0.2*500))]
            if seg_u.size: Uamp.append(np.max(seg_u) - baseline)
        feats.update({
            'ST_dev_mean':   safe_mean(np.array(STdev)),
            'ST_slope_mean': safe_mean(np.array(STslope)),
            'T_amp_mean':    safe_mean(np.array(Tamp)),
            'T_sym_mean':    safe_mean(np.array(Tsym)),
            'U_amp_mean':    safe_mean(np.array(Uamp)),
            'subject_id':    subject_id,
            'study_id':      study_id
        })
        return feats
    except Exception:
        return None


def batch_compute_parallel(data_dir: str, out_csv: str):
    loader = MIMICIVECGLoader(data_dir)
    rec_df = loader.record_list[['subject_id', 'study_id']]
    records = loader.record_list[['subject_id','study_id','path']].to_records(index=False)
    entries = [(int(r.subject_id), int(r.study_id), r.path, data_dir) for r in records]
    n_cpu = max(1, multiprocessing.cpu_count() - 1)
    results = []
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {executor.submit(compute_features_for_entry, e): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing ECG records"):
            res = future.result()
            if res is not None:
                results.append(res)
    df = pd.DataFrame(results)
    # Merge to preserve original order
    merged = rec_df.merge(df, on=['subject_id','study_id'], how='left')
    # Reorder columns: subject_id, study_id first
    cols = ['subject_id','study_id'] + [c for c in merged.columns if c not in ['subject_id','study_id']]
    merged[cols].to_csv(out_csv, index=False)
    print(f"Saved morphology features to {out_csv}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='MIMIC-IV-ECG 根目录')
    parser.add_argument('--out-csv', default='record_list_morphology.csv', help='输出 CSV 文件路径')
    args = parser.parse_args()
    batch_compute_parallel(args.data_dir, args.out_csv)
