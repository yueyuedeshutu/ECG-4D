# ecg_analysis/ecg_processor.py
import os
import numpy as np
import wfdb
import neurokit2 as nk
from tqdm import tqdm
from scipy.fft import fft, fftfreq
import multiprocessing


def read_leads(signal_path):
    record = wfdb.rdrecord(signal_path)
    return {name: record.p_signal[:, i] for i, name in enumerate(record.sig_name)}


def sokolow_lyon_index(leads):
    try:
        v1 = leads.get('V1'); v5 = leads.get('V5'); v6 = leads.get('V6')
        if v1 is None or (v5 is None and v6 is None):
            return np.nan
        s_v1 = abs(np.min(v1))
        r_v5v6 = max(np.max(v5) if v5 is not None else 0,
                      np.max(v6) if v6 is not None else 0)
        return s_v1 + r_v5v6
    except:
        return np.nan


def cornell_voltage_index(leads):
    try:
        avl = leads.get('aVL'); v3 = leads.get('V3')
        if avl is None or v3 is None:
            return np.nan
        r_avl = np.max(avl)
        s_v3 = abs(np.min(v3))
        return r_avl + s_v3
    except:
        return np.nan


def cornell_voltage_product_index(leads, qrs_duration_ms):
    index = cornell_voltage_index(leads)
    try:
        # QRS duration in ms, product in mV·ms
        return index * qrs_duration_ms if not np.isnan(index) else np.nan
    except:
        return np.nan


def detect_noise_types(leads, sampling_rate=500):
    baseline_wander = static_noise = burst_noise = False
    electrode_issue = powerline_noise = False
    for signal in leads.values():
        if np.isnan(signal).any():
            continue
        freqs = fftfreq(len(signal), 1 / sampling_rate)
        spectrum = np.abs(fft(signal))

        # Baseline wander detection (0.05-0.5 Hz)
        if np.mean(spectrum[(freqs > 0.05) & (freqs < 0.5)]) > 0.01 * np.mean(spectrum):
            baseline_wander = True
        # Static noise (flat line) detection in first second
        if np.std(signal[:sampling_rate]) < 0.05:
            static_noise = True
        # Electrode issue (very low amplitude)
        if np.mean(np.abs(signal)) < 0.05:
            electrode_issue = True
        # Powerline interference (around 60 Hz notch frequency)
        if np.mean(spectrum[(freqs > 59) & (freqs < 61)]) > 0.05 * np.mean(spectrum):
            powerline_noise = True
        # Burst noise: sudden large spikes relative to variability
        diffs = np.abs(np.diff(signal))
        if np.any(diffs > 5 * np.std(signal)):
            burst_noise = True

    total_issues = sum([baseline_wander, static_noise, burst_noise, electrode_issue, powerline_noise])
    quality_score = max(round(1 - total_issues * 0.2, 2), 0)
    return baseline_wander, static_noise, burst_noise, electrode_issue, powerline_noise, quality_score


def extract_rr_interval(signal_path, sampling_rate=500):
    try:
        record = wfdb.rdrecord(signal_path)
        sig = record.p_signal
        # Use Lead II if available
        lead_sig = sig[:, 1] if sig.shape[1] > 1 else sig[:, 0]
        clean = nk.ecg_clean(lead_sig, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(clean, sampling_rate=sampling_rate)
        locs = rpeaks['ECG_R_Peaks']
        rr_samples = np.diff(locs)
        if len(rr_samples) > 0:
            # Convert to milliseconds
            rr_ms = np.mean(rr_samples) * (1000.0 / sampling_rate)
            return int(rr_ms)
        return 0
    except:
        return 0


def process_record(path, base_dir, qrs_duration_ms):
    full_path = os.path.join(base_dir, path)
    leads = read_leads(full_path)
    rr_interval = extract_rr_interval(full_path)
    sokolow = sokolow_lyon_index(leads)
    cornell = cornell_voltage_index(leads)
    cornell_prod = cornell_voltage_product_index(leads, qrs_duration_ms)
    noise = detect_noise_types(leads)

    return {
        'RR_Interval': rr_interval,
        'Sokolow_Lyon_Index': sokolow,
        'Cornell_Voltage_Index': cornell,
        'Cornell_Product_Index': cornell_prod,
        'Baseline_Wander': noise[0],
        'Static_Noise': noise[1],
        'Burst_Noise': noise[2],
        'Electrode_Issue': noise[3],
        'Powerline_Interference': noise[4],
        'Signal_Quality': noise[5],
    }


def _process_worker(args):  # helper for multiprocessing
    base_dir, path, qrs_duration_ms = args
    return process_record(path, base_dir, qrs_duration_ms)


def batch_process(base_dir, record_paths, qrs_durations):
    """
    Parallel batch processing using all available CPUs.
    """
    args = [(base_dir, p, q) for p, q in zip(record_paths, qrs_durations)]
    nprocs = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=nprocs) as pool:
        results = list(tqdm(pool.imap(_process_worker, args), total=len(args)))
    return [res for res in results if res is not None]
