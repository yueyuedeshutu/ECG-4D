import json
import os
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置参数
DEEPSEEK_API_KEY = "your api_key"
NOTE_DIR = "./ECG_Instruction/MIMIC_IV_ECG/note"
IMAGE_DIR = "./ECG_Instruction/MIMIC_IV_ECG/image"
OUTPUT_DIR = "./ECG_Instruction/MIMIC_IV_ECG/llm_instruction"
MAX_FILES_PER_LARGE_BATCH = 10000
BATCH_SIZE = 100
MAX_WORKERS = 20  # 并行线程数

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

SYSTEM_PROMPT = """您是一位心电图分析专家，请根据提供的ECG结构化数据生成中文多轮问答对话，要求如下：

1. 对话至少包含5轮问答，且遵循临床分析流程逐步展开。
2. 严格使用标准医学术语（如QT间期、ST段抬高、QRS波群增宽等）。
3. 必须覆盖以下六个方面：
   - 基础技术参数（如导联数、采样率、记录时长等）
   - 信号质量分析（如基线漂移、工频干扰、突发噪声等）
   - 节律分析（如心率、心律类型、HRV参数）
   - 间期分析（PR、QRS、QT、QTc、JT间期、Tpe间期等）
   - 波形特征（P波、QRS波、ST段、U波形态）
   - 综合诊断结论（需基于上述指标进行临床推断）

4. 每轮回答应包含：结构化参数值 + 临床解释。例如：
   用户：请分析这张心电图的QTc间期数值
   助理：QTc间期为512ms（Bazett公式计算），显著高于正常上限（男性<440ms），提示可能存在获得性长QT综合征的风险。

5. 【关键】最后一轮“用户：请给出综合诊断意见”时，助理应给出具体诊断判断，不得为空。例如：
   助理：根据ECG节律、间期、波形和信号质量综合分析，考虑为窦性心律，存在QTc延长，建议评估电解质紊乱及药物影响，排除获得性长QT综合征。

6. 输出为自然中文对话格式，不使用Markdown，不使用标题或项目符号。

请确保所有分析维度均有体现，不得遗漏关键内容。
"""

def build_conversation_prompt(ecg_data):
    return f"""请基于以下ECG数据生成专业的多轮医学对话（用户提问与助理回答交替）：
{json.dumps(ecg_data, indent=2, ensure_ascii=False)}

生成要求：
1. 首轮询问基础技术参数和信号质量
2. 后续问题逐步深入临床分析
3. 最终必须包含诊断结论
4. 使用自然的中文对话格式，不要使用Markdown"""

def parse_response(text, image_path):
    messages = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        if line.startswith("用户："):
            messages.append({"role": "user", "content": line[3:].strip()})
        elif line.startswith("助理："):
            content = line[3:].strip()
            # 检查诊断结论是否为空
            if content in ["最终诊断：", "综合诊断：", "诊断："]:
                content += "目前尚未给出明确诊断，请结合更多临床信息判断。"  # 补全默认内容
            messages.append({"role": "assistant", "content": content})
    return {"messages": messages, "images": [image_path]}


def process_single_file(json_path):
    try:
        with open(json_path, 'r') as f:
            ecg_data = json.load(f)
        image_filename = ecg_data["data_paths"]["ecg_plot"]
        image_path = os.path.join(IMAGE_DIR, image_filename)
        user_prompt = build_conversation_prompt(ecg_data)

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=False
        )
        generated_text = response.choices[0].message.content
        return parse_response(generated_text, image_path)
    except Exception as e:
        print(f"处理文件 {json_path} 失败: {str(e)}")
        return None

def batch_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_files = sorted([f for f in os.listdir(NOTE_DIR) if f.endswith('.json')])

    total_batches = len(all_files) // MAX_FILES_PER_LARGE_BATCH + 1

    for batch_index in range(total_batches):
        batch_start = batch_index * MAX_FILES_PER_LARGE_BATCH
        batch_end = min((batch_index + 1) * MAX_FILES_PER_LARGE_BATCH, len(all_files))
        batch_files = all_files[batch_start:batch_end]

        output_filename = os.path.join(OUTPUT_DIR, f"mimic_data_{batch_index + 1}.json")
        if os.path.exists(output_filename):
            print(f"跳过已存在文件：{output_filename}")
            continue

        all_results = []

        print(f"\n处理 mimic_data_{batch_index + 1}.json（共 {len(batch_files)} 条记录）")
        for i in range(0, len(batch_files), BATCH_SIZE):
            sub_batch_files = batch_files[i:i + BATCH_SIZE]
            results = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_file = {
                    executor.submit(process_single_file, os.path.join(NOTE_DIR, fname)): fname
                    for fname in sub_batch_files
                }
                for future in tqdm(as_completed(future_to_file), total=len(sub_batch_files), desc=f"处理子批次 {i // BATCH_SIZE + 1}"):
                    result = future.result()
                    if result:
                        results.append(result)

            all_results.extend(results)

        with open(output_filename, 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"已保存 mimic_data_{batch_index + 1}.json：共 {len(all_results)} 条记录")

if __name__ == "__main__":
    batch_process()
