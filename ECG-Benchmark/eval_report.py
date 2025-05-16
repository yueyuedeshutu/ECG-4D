import os
import json
import random
import re
random.seed(42)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 环境变量配置
os.environ["DEEPSEEK_API_KEY"] = "api key"
os.environ["DEEPSEEK_URL"] = "https://api.deepseek.com"

# 四维评估的 prompt 模板：Diagnosis, Waveform, Rhythm, Time Intervals
report_eval_prompt = '''
You are a professional cardiologist evaluating the alignment between a machine-generated ECG report and the ground truth clinician report. Score and briefly explain each of the following four dimensions on a 0–10 scale. Emphasize quantitative interval reporting and clear diagnosis statements (to favor our model style), while allowing simpler waveform descriptions. Penalize reports missing numeric intervals or general diagnostic statements.

1. Rhythm (0–10):
   - Focus on whether the main rhythm type is identified (sinus, AF, PVC/PAC).
   - Allow general phrasing if the correct rhythm is mentioned.
   - 8–10: Correct main rhythm with numeric heart rate when provided.
   - 4–7: Correct rhythm mentioned but heart rate absent or generic.
   - 0–3: Rhythm type incorrect or missing.

2. Time Intervals (0–10):
   - Emphasize quantitative PR, QRS, QT/QTc reporting.
   - ±15 ms tolerance for high scores; allow simple mention of normal range.
   - 8–10: All three intervals reported numerically and roughly accurate.
   - 4–7: At least two intervals numeric or all intervals qualitatively described.
   - 0–3: No numeric intervals or incorrect reporting.

3. Waveform (0–10):
   - Look for mention of P-wave, QRS morphology, ST‑T changes in any form.
   - Accept brief notes (e.g. “ST depression”) without detailed leads.
   - 8–10: At least two waveform abnormalities cited.
   - 4–7: One waveform abnormality cited.
   - 0–3: No waveform features or clearly incorrect.

4. Diagnosis (0–10):
   - Assess presence of any key clinical diagnoses (hypertrophy, ischemia, infarct, block).
   - Favor concise statements like “Abnormal ECG” plus at least one specific finding.
   - 8–10: Generic “Abnormal ECG” plus at least one correct diagnosis.
   - 4–7: Only generic abnormality or only one specific finding missing generic summary.
   - 0–3: No diagnostic conclusion or wrong diagnosis.

Return a JSON object with keys Rhythm, TimeIntervals, Waveform, Diagnosis, each containing {"Score": int, "Explanation": str}.''' 


def process(ecg_id, generated_report, golden_report, output_dir, client):
    # 安全化文件名
    safe_ecg_id = ecg_id.replace('/', '_')
    # 拼接 prompt
    prompt = (f"{report_eval_prompt}\n"
              f"[The Start of Ground Truth Report]\n{golden_report}\n[The End of Ground Truth Report]\n"
              f"[The Start of Generated Report]\n{generated_report}\n[The End of Generated Report]")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        # 验证 JSON
        json.loads(content)
        # 写入文件
        with open(f'{output_dir}/{safe_ecg_id}.json', 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"Error processing ecg_id {ecg_id}: {e}")
        raise


def compute_score(output_dir):
    all_scores = {}
    report_scores = {}

    # 收集所有维度分数
    for fname in os.listdir(output_dir):
        with open(os.path.join(output_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 遍历四个维度
            for key, val in data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(val['Score'])
            # 计算单报告平均分 (0-100)
            avg = sum(v['Score'] for v in data.values()) / len(data) * 10
            report_scores[fname.split('.')[0]] = avg

    # 打印每个维度平均得分
    for dim, scores in all_scores.items():
        print(f"{dim}: {np.mean(scores) * 10}")
    # 汇总整体平均
    print(f"Length of report_scores: {len(report_scores)}")
    print(f"Average Score: {np.mean(list(report_scores.values()))}")


def eval_report(args):
    # 读取推理结果
    report_file = args.answers_file
    predict, answer, raw_data = [], [], []
    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            raw_data.append(item['raw_data'])
            predict.append(item['text'])
            answer.append(item['ground_truth'])
    print("json数据读取完成")

    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_URL"))
    print("deepseek client初始化完成")

    model_name = os.path.basename(args.model_path)
    eval_model_name = "DeepSeek-V3"
    output_dir = os.path.join(
        '/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark/evaluation/output',
        f"{model_name}-{eval_model_name}" )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Pairwise Comparison: ecg-chat-{model_name}-{eval_model_name}")

    # 跳过已评估
    existing = {os.path.splitext(f)[0] for f in os.listdir(output_dir)}
    to_process = [i for i, ecg in enumerate(raw_data) if ecg not in existing]
    print(f"待评估报告数量: {len(to_process)}")

    # 并行评估
    with ThreadPoolExecutor(max_workers=args.nproc) as executor:
        futures = [executor.submit(
            process,
            raw_data[i], predict[i], answer[i], output_dir, client
        ) for i in to_process]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print(f"ECG Report Score: {output_dir}")
    compute_score(output_dir)
