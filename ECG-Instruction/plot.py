import wfdb
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
import multiprocessing
import os
import math

class ECGPlotter:

    def __init__(self, mimic_root: str, output_dir: str, process_id: int = 0):
        """
        参数:
            mimic_root: MIMIC-IV-ECG数据集根目录
            output_dir: 图像输出目录
            process_id: 进程标识（用于区分不同进程）
        """
        self.mimic_root = Path(mimic_root)
        self.output_dir = Path(output_dir)
        self.process_id = process_id
        self._validate_paths()

        # 设置进程专属的完成记录文件
        self.completed_records_file = self.output_dir / f'completed_records_p{process_id}.txt'
        self.completed_indices = self._load_completed_indices()

        # 配置进程专属的日志
        logging.basicConfig(
            filename=self.output_dir / f'plotting_errors_p{process_id}.log',
            level=logging.ERROR,
            format='%(asctime)s - %(message)s',
            force=True  # 覆盖之前的配置
        )

    def _validate_paths(self):
        """校验路径有效性"""
        if not self.mimic_root.exists():
            raise FileNotFoundError(f"MIMIC-IV根目录不存在: {self.mimic_root}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_completed_indices(self) -> set:
        """加载已完成的记录编号"""
        if self.completed_records_file.exists():
            with open(self.completed_records_file, 'r') as f:
                return set(int(line.split(',')[0]) for line in f if line.strip())
        return set()

    def generate_ecg_plots(self, record_list_path: str, start_idx: int = 0, max_count: int = None, disable_pbar: bool = True):
        """
        批量生成心电图图像（支持进度条控制）
        
        参数:
            record_list_path: record_list.csv路径
            start_idx: 起始序号
            max_count: 最大处理数量
            disable_pbar: 是否禁用进度条
        """
        records = self._load_records(record_list_path)
        records = records[start_idx : start_idx+max_count] if max_count else records[start_idx:]

        pbar = tqdm(
            enumerate(records, start=start_idx),
            total=len(records),
            desc=f"Process {self.process_id}",
            unit="records",
            disable=disable_pbar
        )

        for global_idx, rel_path in pbar:
            if global_idx in self.completed_indices:
                continue
            try:
                self._process_single_record(rel_path, global_idx)
                with open(self.completed_records_file, 'a') as f:
                    f.write(f"{global_idx},{rel_path}\n")
            except Exception as e:
                msg = f"进程{self.process_id} 处理失败 [{rel_path}]: {str(e)}"
                logging.error(msg, exc_info=True)
                pbar.write(msg)

    def _load_records(self, csv_path: str) -> list:
        """从CSV加载记录列表"""
        df = pd.read_csv(csv_path)
        return df['path'].tolist()

    def _process_single_record(self, rel_path: str, global_idx: int):
        """处理单条记录"""
        full_path = self.mimic_root / rel_path
        record = wfdb.rdrecord(str(full_path))
        self._plot_and_save(record, global_idx)

    def _plot_and_save(self, record, idx: int):
        """执行绘图并保存（保持原始画质参数）"""
        output_path = self.output_dir / f"{idx:07d}.png"

        plt.figure(figsize=(24, 18), dpi=100)
        plt.suptitle(f"{idx:07d}", y=0.99, fontsize=12)

        wfdb.plot_wfdb(
            record=record,
            title=f"Record {idx:07d}",
            ecg_grids='all',
            figsize=(24, 18),
            return_fig=False
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()

def worker_process(args):
    """工作进程执行函数"""
    mimic_root, output_dir, record_list, start_idx, max_count, process_id = args
    plotter = ECGPlotter(
        mimic_root=mimic_root,
        output_dir=output_dir,
        process_id=process_id
    )
    plotter.generate_ecg_plots(
        record_list_path=record_list,
        start_idx=start_idx,
        max_count=max_count,
        disable_pbar=False  # 允许显示进度条
    )

def main():
    # 配置参数
    mimic_root = "./mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
    record_list = "./mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/record_list.csv"
    output_dir = "./data/image"
    
    # 并行配置
    num_processes = multiprocessing.cpu_count()  # 根据CPU核心数自动设置
    df = pd.read_csv(record_list)
    total_records = len(df)
    records_per_process = math.ceil(total_records / num_processes)

    print(f"✅ ECG 并行绘图任务启动！")
    print(f"📁 MIMIC根目录: {mimic_root}")
    print(f"📄 总记录数: {total_records}")
    print(f"🚀 使用进程数: {num_processes}")
    print(f"📤 输出目录: {output_dir}")

    # 准备进程参数
    tasks = []
    for pid in range(num_processes):
        start = pid * records_per_process
        count = min(records_per_process, total_records - start)
        if count <= 0:
            break
        tasks.append((
            mimic_root,
            output_dir,
            record_list,
            start,
            count,
            pid  # 进程ID
        ))

    # 启动进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(worker_process, tasks)

    print(f"✅ 全部任务完成！图像保存在: {output_dir}")

if __name__ == "__main__":
    main()
