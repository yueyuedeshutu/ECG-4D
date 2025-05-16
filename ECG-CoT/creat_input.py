import os
import json
import pandas as pd
from textwrap import dedent

def format_value(value, unit="", precision=2):
    """格式化数值显示"""
    if isinstance(value, float):
        return f"{round(value, precision)}{unit}"
    if isinstance(value, bool):
        return "存在" if value else "未见"
    if value is None:
        return "未观察到"
    return f"{value}{unit}"

def generate_clinical_report(data):
    """生成符合临床思维的诊断分析文本"""
    meta = data["metadata"]
    signal_quality = data["signal_quality"]
    metrics = data["diagnostic_metrics"]
    conclusion = data["diagnostic_conclusion"]
    
    # 信号质量分析
    sq_analysis = dedent(f"""\
    1）信号质量分析：
    - 基线漂移：{format_value(signal_quality["Baseline_Wander"])}
    - 工频干扰：{format_value(signal_quality["Powerline_Interference"])}
    - 突发噪声：{format_value(signal_quality["Burst_Noise"])}
    - 总体信号质量评分：{format_value(signal_quality["Signal_Quality"], "", 1)}/1.0
    """)
    
    # 心率分析
    hr = metrics["rhythm_metrics"]["heart_rate"]
    hrv = hr["HRV"]
    hr_analysis = dedent(f"""\
    2）心率分析：
    - 平均心率：{format_value(hr["mean_HR"], "bpm")}
    - 心率变异性：
      * SDNN：{format_value(hrv["time_domain"]["SDNN"], "ms")} 
      * RMSSD：{format_value(hrv["time_domain"]["RMSSD"], "ms")}
      * pNN50：{format_value(hrv["time_domain"]["pNN50"], "%")}
      * LF/HF 比值：{format_value(hrv["frequency_domain"]["LF_HF_ratio"], "", 4)}
    """)
    
    # 间期分析（新增QT相关指标）
    interval = metrics["interval_metrics"]
    interval_analysis = dedent(f"""\
    3）间期分析：
    - PR间期：{format_value(interval["PR_interval"], "ms")}（正常值120-200ms）
    - QT间期：{format_value(interval["QT_interval"], "ms")}
    - QT离散度：{format_value(interval["QT_dispersion"], "ms")}
    - QTc间期：{format_value(interval["QTc_interval"], "ms")}（Bazett公式，正常<440ms）
    - Tpe间期：{format_value(interval["Tpe_interval"], "ms")}
    - JT间期：{format_value(interval["JT_interval"], "ms")}
    """)
    
    # 波形特征分析（新增波形参数）
    wave = metrics["morphology_metrics"]
    wave_analysis = dedent(f"""\
    4）波形特征分析：
    P波：
    - 振幅：{format_value(wave["P_wave"]["amplitude"], "mV", 3)}
    - 时限：{format_value(wave["P_wave"]["duration"], "ms")}
    - 离散度：{format_value(wave["P_wave"]["dispersion"], "ms")}
    - 切迹：{format_value(wave["P_wave"]["notching"])}
    - 起始点：{format_value(wave["P_wave"]["p_onset"], "ms")}
    - 终止点：{format_value(wave["P_wave"]["p_end"], "ms")}
    
    QRS波群：
    - QRS波时限：{format_value(wave["QRS_complex"]["duration"], "ms")}
    - R波振幅（II导联）：{format_value(wave["QRS_complex"]["R_amplitude"], "mV", 3)}
    - S波深度（V2导联）：{format_value(wave["QRS_complex"]["S_depth"], "mV", 3)}
    - QRS波起始点：{format_value(wave["QRS_complex"]["qrs_onset"], "ms")}
    - QRS波终止点：{format_value(wave["QRS_complex"]["qrs_end"], "ms")}
    - R/S比值（V1导联）：{format_value(wave["QRS_complex"]["R/S_ratio"]["R/S_V1"], "", 2)}
    - 病理性Q波：{format_value(wave["QRS_complex"]["pathological_q_wave"])}
    - 碎裂QRS波：{format_value(wave["QRS_complex"]["fragmented_qrs"])}
    
    ST-T段：
    - ST段偏移：{format_value(wave["ST_T_wave"]["ST_deviation"], "mm", 1)}
    - ST段斜率：{format_value(wave["ST_T_wave"]["ST_segment_slope"], "μV/s", 1)}
    - T波振幅：{format_value(wave["ST_T_wave"]["T_amplitude"], "mV", 3)}
    - T波对称性：{format_value(wave["ST_T_wave"]["T_symmetry"], "", 2)}
    - T波终止点：{format_value(wave["ST_T_wave"]["t_end"], "ms")}
    
    U波：
    - 振幅：{format_value(wave["U_wave"]["amplitude"], "mV", 3)}
    """)
    
    # 向量与结构分析（新增电轴参数）
    vector = metrics["vector_metrics"]
    struct = metrics["structural_metrics"]
    vector_analysis = dedent(f"""\
    5）向量指标分析：
    额面电轴分析：
    - QRS轴：{format_value(vector["frontal_axis"]["QRS_axis"], "°")}
    - T轴：{format_value(vector["frontal_axis"]["T_axis"], "°")}
    - P轴：{format_value(vector["frontal_axis"]["P_axis"], "°")}
    - QRS-T夹角：{format_value(vector["frontal_axis"]["QRS_T_angle"], "°")}
    
    水平面电轴分析：
    - R波递增区导联：{format_value(vector["horizontal_axis"]["RS_transition_lead"])}
    - R波上升斜率：{format_value(vector["horizontal_axis"]["R_wave_slope"], "mV/ms", 4)}
    
    6）结构性指标：
    - Sokolow-Lyon指数：{format_value(struct["Sokolow_Lyon"], "mV")}
    - Cornell电压：{format_value(struct["Cornell_Voltage"], "mV")}
    - Cornell乘积：{format_value(struct["Cornell_Product"], "mm·ms")}
    """)
    
    # 诊断结论
    diag_conclusion = dedent(f"""\
    7）最终诊断结果：
    主要诊断：{conclusion["primary_diagnosis"]}
    次要诊断：{conclusion["secondary_diagnosis"]}
    补充诊断：{conclusion["supplementary_description"]},{chr(10).join(['* '+item for item in conclusion["extended_annotations"]])}
    """)
    
    # 整合报告
    full_report = (
      f"您是一名资深心内科医生，您的任务是基于给定的 12 导联系统化心电图特征，自动生成符合临床思维的诊断推理链（Chain-of-Thought），并输出最终诊断结论。\n\n"
      f"{sq_analysis}\n{hr_analysis}\n{interval_analysis}\n"
      f"{wave_analysis}\n{vector_analysis}\n{diag_conclusion}\n\n"
      "请按照以下“诊断流程”分步输出完整的推理链与结论：\n\n"
      "【一】诊断流程\n"
      "1. 信号质量验证：检查基线漂移、工频干扰、突发噪声，若不可靠标注“不可靠”并推荐改进措施。\n"
      "2. 心率与变异性分析：计算平均心率及 HRV（SDNN、RMSSD、pNN50、LF/HF），识别异常并量化。\n"
      "3. 时间间期：测量 PR、QT、QTc（多公式对比）、Tpe、JT等时间间期指标，并对每个指标进行详细分析。\n"
      "4. 波形特征：评估 P 波、QRS、ST-T、U 波形态的各项指标，识别相关异常。\n"
      "5. 向量指标：分析额面/水平面电轴的相关指标，判断心脏的收缩功能是否正常。\n"
      "6. 结构性指标：分析Sokolow-Lyon、Cornell 电压/乘积指标，评估心室肥厚等。\n"
      "7. 综合诊断：整合上述各项，给出主要/次要/补充诊断，并附置信度或文献引用。\n\n"
      "【二】约束条件\n"
      "1）动态知识检索\n"
      "   - 在每个关键决策节点（如 QTc 校正、左室肥大标准判断等）调用最新的临床规则知识库（Knowledge Graph）进行校验，并引用规则名称与最新发布年份。\n"
      "   - 示例：\n"
      "     “根据 2023 年 AHA 心电图指南，QTc 正常范围＜440ms” 或 “依据最新文献（PMID: 12345678）LF/HF 正常值在 0.5–2.0”。\n\n"
      "2）不确定性量化\n"
      "   - 对所有关键数值（如 SDNN、QTc、ST 偏移等）计算并展示置信度评分或置信区间。\n"
      "   - 输出格式须包含数值、参考范围、及“置信度: xx%”或“95% CI [a–b]”。\n"
      "   - 示例：\n"
      "     “SDNN 35 ms（↓，95% CI [30–40]）[置信度: 90%]”\n\n"
      "3）矛盾检测与回溯校正\n"
      "   - 若出现内部逻辑冲突（如 Cornell 电压正常而 Sokolow-Lyon 指数增高），自动标注冲突源。\n"
      "   - 提供两种对立结论及其概率分布，并建议进一步检查路径。\n"
      "   - 示例：\n"
      "     “Cornell 18 mV（正常） vs S-L 指数 4.5 mV（增高）[冲突点] →\n"
      "       • 左室肥厚可能性 70%  • 无左室肥厚可能性 30%，建议影像学验证。”\n\n"
      "请严格按照上述“诊断流程”生成完整的临床推理链，并在“最终诊断”部分附上主要诊断、次要诊断、补充说明及相应置信度或规则来源标注。\n"
    )
    
    return full_report

def main():
    input_dir = "/gpfsdata/home/liuxinyue/ECG/10.COT/normal/note_normal"
    output_csv = "/gpfsdata/home/liuxinyue/ECG/10.COT/build_cot/input_final_normal.csv"
    
    records = []
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            report = generate_clinical_report(data)
            records.append({
                "note_path": filename,
                "input": report
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    pd.DataFrame(records).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"生成{len(records)}条记录，保存至{output_csv}")

if __name__ == "__main__":
    main()
