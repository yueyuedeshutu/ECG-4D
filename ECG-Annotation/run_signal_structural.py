import os
import pandas as pd
import argparse
from ecg_analysis import batch_process
import warnings
warnings.filterwarnings("ignore")

def main(data_dir):
    # Sampling rate for ECG signals (Hz)
    sampling_rate = 500

    # Load record list and metadata
    df = pd.read_csv(os.path.join(data_dir, "record_list.csv"))
    meta = pd.read_csv(os.path.join(data_dir, "machine_measurements.csv"), low_memory=False)

    # Ensure numeric QRS onset/end (sample indices)
    for col in ['qrs_onset', 'qrs_end']:
        meta[col] = pd.to_numeric(meta[col], errors='coerce')
    # Compute QRS duration in milliseconds
    meta['qrs_duration_ms'] = meta['qrs_end'] - meta['qrs_onset']
    meta['qrs_duration_ms'] = meta['qrs_duration_ms'].round().astype(int)

    # Merge metadata into record list
    meta_sel = meta[['subject_id', 'study_id', 'qrs_duration_ms']]
    df = df.merge(meta_sel, on=['subject_id', 'study_id'], how='left')

    print("Starting ECG feature extraction with parallel processing...")
    # Batch process with precomputed QRS durations
    qrs_list = df['qrs_duration_ms'].fillna(0).tolist()
    results = batch_process(data_dir, df['path'].tolist(), qrs_list)

    # Append results and save
    for key in results[0]:
        df[key] = [r[key] for r in results]
    out_file = os.path.join(data_dir, 'record_list_structural.csv')
    df.to_csv(out_file, index=False)
    print(f"Finished. Parallel results saved to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to directory containing record_list.csv and ECG files.")
    args = parser.parse_args()
    main(args.data_dir)
