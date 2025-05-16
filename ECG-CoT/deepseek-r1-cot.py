import pandas as pd
from openai import OpenAI
import time
import logging
import os
from tqdm import tqdm
import csv

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_processing.log'),
        logging.StreamHandler()
    ]
)

class ECGProcessor:
    def __init__(self, input_file, output_file, api_key):
        self.input_file = input_file
        self.output_file = output_file
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 初始化已处理记录
        self.processed_records = self._load_processed_records()
        
    def _load_processed_records(self):
        """加载已处理的记录"""
        if os.path.exists(self.output_file):
            try:
                df = pd.read_csv(self.output_file)
                return set(df['note_path'].tolist())
            except:
                return set()
        return set()

    def _save_result(self, result):
        """实时保存单条结果"""
        file_exists = os.path.isfile(self.output_file)
        mode = 'a' if file_exists else 'w'
        
        with open(self.output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['note_path', 'reasoning', 'output'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

    def process_records(self):
        """主处理流程"""
        try:
            df_input = pd.read_csv(self.input_file)
        except Exception as e:
            logging.error(f"读取输入文件失败：{str(e)}")
            return

        total = len(df_input)
        success_count = 0
        error_count = 0

        with tqdm(total=total, desc="处理进度") as pbar:
            for index, row in df_input.iterrows():
                note_path = row['note_path']
                
                # 跳过已处理记录
                if note_path in self.processed_records:
                    pbar.update(1)
                    continue
                
                try:
                    # API调用
                    stream_response = self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": row['input']}],
                        stream=True
                    )

                    # 处理流式响应
                    reasoning, content = "", ""
                    for chunk in stream_response:
                        if chunk.choices[0].delta.reasoning_content:
                            reasoning += chunk.choices[0].delta.reasoning_content
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content

                    # 保存结果
                    result = {
                        'note_path': note_path,
                        'reasoning': reasoning.strip(),
                        'output': content.strip()
                    }
                    self._save_result(result)
                    self.processed_records.add(note_path)
                    success_count += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': error_count,
                        '当前': note_path
                    })
                    pbar.update(1)

                    # 请求间隔
                    time.sleep(0.3)

                except Exception as e:
                    error_count += 1
                    logging.error(f"处理失败 {note_path}: {str(e)}")
                    continue

        logging.info(f"\n处理完成！成功: {success_count}, 失败: {error_count}")

if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "./build_cot/input_final_normal.csv"
    OUTPUT_FILE = "./build_cot/output_final_normal.csv"
    API_KEY = "api_key"  # 替换为实际API密钥

    processor = ECGProcessor(INPUT_FILE, OUTPUT_FILE, API_KEY)
    processor.process_records()
