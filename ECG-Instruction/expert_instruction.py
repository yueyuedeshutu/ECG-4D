#from original
import os
import json
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Helper for safe formatting of numeric values
def safe_fmt(val):
    try:
        return f"{val:.2f}"
    except (TypeError, ValueError):
        return "0.00"

# 输入输出路径配置
csv_path    = '/gpfsdata/home/liuxinyue/ECG/9.instruction_youhua/note_split/note_paths_split_4.csv'
output_dir  = '/gpfsdata/home/liuxinyue/ECG/9.instruction_youhua/instruction_split'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mimic_data_part_4.json')
error_log   = os.path.join(output_dir, 'errors.log')

def process_file(json_path):
    record_id = os.path.splitext(os.path.basename(json_path))[0]
    try:
        # 直接用 json_path 打开文件
        with open(json_path, 'r', encoding='utf-8') as f:
            note = json.load(f)
        
        # 提取信号质量并生成描述字符串
        sq = note.get('signal_quality', {})
        flags = [
            '有基线漂移'     if sq.get('Baseline_Wander')       else '无基线漂移',
            '有静电噪声'     if sq.get('Static_Noise')          else '无静电噪声',
            '有突发噪声'     if sq.get('Burst_Noise')           else '无突发噪声',
            '有电极接触不良' if sq.get('Electrode_Issue')       else '无电极接触不良',
            '有工频干扰'     if sq.get('Powerline_Interference') else '无工频干扰'
        ]
        sq_score = safe_fmt(sq.get("Signal_Quality"))
        quality_desc = '，'.join(flags) + f'，综合信号质量评分为{sq_score}'

        # 提取节律指标
        hr_raw   = note['diagnostic_metrics']['rhythm_metrics']['heart_rate']['mean_HR']
        lfhf_raw = note['diagnostic_metrics']['rhythm_metrics']['heart_rate']['HRV']['frequency_domain']['LF_HF_ratio']
        hr   = safe_fmt(hr_raw)
        lfhf = safe_fmt(lfhf_raw)

        # 提取各间期指标
        intervals = note['diagnostic_metrics']['interval_metrics']
        rr   = safe_fmt(intervals['RR_interval']['mean'])
        pr   = safe_fmt(intervals['PR_interval'])
        qrs  = safe_fmt(intervals['QRS_duration'])
        qt   = safe_fmt(intervals['QT_interval'])
        qtc  = safe_fmt(intervals['QTc_interval'])
        qtd  = safe_fmt(intervals['QT_dispersion'])
        jt   = safe_fmt(intervals['JT_interval'])
        tpe  = safe_fmt(intervals['Tpe_interval'])

        # 提取形态指标
        morph   = note['diagnostic_metrics']['morphology_metrics']
        p_wave  = morph['P_wave']
        p_amp   = safe_fmt(p_wave.get('amplitude'))
        p_dur   = safe_fmt(p_wave.get('duration'))
        p_disp  = safe_fmt(p_wave.get('dispersion'))
        p_notch = p_wave.get('notching', False)

        qrs_c = morph['QRS_complex']
        r_amp   = safe_fmt(qrs_c.get('R_amplitude'))
        s_dep   = safe_fmt(qrs_c.get('S_depth'))
        path_q  = qrs_c.get('pathological_q_wave', False)
        frag_q  = qrs_c.get('fragmented_qrs', False)
        rsr     = qrs_c.get('R/S_ratio', {})
        rs_v1   = safe_fmt(rsr.get('R/S_V1'))
        rs_v2   = safe_fmt(rsr.get('R/S_V2'))
        rs_v3   = safe_fmt(rsr.get('R/S_V3'))
        rs_v4   = safe_fmt(rsr.get('R/S_V4'))
        rs_v5   = safe_fmt(rsr.get('R/S_V5'))
        rs_v6   = safe_fmt(rsr.get('R/S_V6'))

        stt = morph['ST_T_wave']
        st_dev = safe_fmt(stt.get('ST_deviation'))
        st_slope = safe_fmt(stt.get('ST_segment_slope'))
        t_amp = safe_fmt(stt.get('T_amplitude'))
        t_sym = safe_fmt(stt.get('T_symmetry'))

        u_wave = morph['U_wave']
        u_amp  = safe_fmt(u_wave.get('amplitude'))

        # 提取向量指标
        frontal   = note['diagnostic_metrics']['vector_metrics']['frontal_axis']
        qrs_ax    = safe_fmt(frontal.get('QRS_axis'))
        t_ax      = safe_fmt(frontal.get('T_axis'))
        qrst_ang  = safe_fmt(frontal.get('QRS_T_angle'))

        horizontal = note['diagnostic_metrics']['vector_metrics']['horizontal_axis']
        rs_lead    = horizontal.get('RS_transition_lead', '')
        r_slope    = safe_fmt(horizontal.get('R_wave_slope'))

        # 提取结构化指数
        struct = note['diagnostic_metrics']['structural_metrics']
        sok   = safe_fmt(struct.get('Sokolow_Lyon'))
        cornV = safe_fmt(struct.get('Cornell_Voltage'))
        cornP = safe_fmt(struct.get('Cornell_Product'))

        # 提取诊断结论
        concl = note.get('diagnostic_conclusion', {}) 
        ext_ann = '、'.join(concl.get('extended_annotations', [])) or '暂无'

        # 构建 LLama Factory 样本结构
        messages = [
            {"role": "user",    "content": "<image>请描述这张图像的图像特征及基础参数"},
            {"role": "assistant", "content": (
                f"本图为标准12导联心电图，包含肢体导联（I、II、III、aVR、aVL、aVF）及胸导联（V1-V6）。"
                f"该心电图的采样率为500Hz（时间分辨率2ms），记录时长10秒，{quality_desc}。"
            )},
            {"role": "user",    "content": "请对这张心电图图像的心率（HR）及心率变异性（HRV）指标进行详细分析"},
            {"role": "assistant", "content": (
                f"心率，即心搏频率，是单位时间内心脏搏动的次数，一般指每分钟的心跳次数，本例平均心率为{hr}次/分。"
                f"心率变异性是通过分析窦性心律逐次心搏间期的差异，评估自主神经系统对心脏节律的调控功能，进而反映心脏自律性及身体机能状态，本例中频域指标LF/HF比值为{lfhf}。"
            )},
            {"role": "user",    "content": "请对这张心电图图像的各间期指标进行详细分析，包括RR间期、PR间期、QRS时限、QT间期、QTc间期、QT离散度、JT间期和Tp-e间期"},
            {"role": "assistant", "content": (
                f"RR间期是相邻两个QRS波群中R波之间的时间间隔，反映心室搏动的周期，本例中的RR间期为{rr}ms。"
                f"PR间期是从P波开始到QRS波群开始的时间间隔，指的是从心房开始除极到心室开始除极的时间，反映了心房与心室之间的电信号传导速度，本例中的PR间期为{pr}ms。"
                f"QRS时限是自QRS波群起点至QRS波群终点的时间，反映了左、右心室除极过程中的电位和时间变化，本例中的QRS时限为{qrs}ms。"
                f"QT间期包括心室除极和复极激动时间，代表心室去极化和复极化过程的总时程，为自QRS波的起点至T波的终点所占的时间，本例中的QT间期为{qt}ms；"
                f"QTc间期是按心率校正的QT间期，是反映心脏去极化和复极作用的指标，本例中的QTc间期为{qtc}ms；"
                f"QT离散度指的是同步记录的12导联体表心电图中最长QT间期与最短QT间期的差值，本例中的QT离散度为{qtd}ms；"
                f"JT间期是心室复极的纯时间，即QRS终点（J点）至T波终点的时间，本例中的JT间期为{jt}ms；"
                f"Tp-e间期是指心电图T波顶点到T波终点之间的时间间期，是QT间期的终末组成部分，能够代表在绝大部分心室肌复极后小部分心室肌间复极的离散度，本例中的Tp-e间期为{tpe}ms。"
            )},
            {"role": "user",    "content": "请对这张心电图图像的各波形指标进行详细分析，包括P波、QRS波群、ST段、T波和U波"},
            {"role": "assistant", "content": (
                f"P波是心房除极波，代表左右二心房的激动。P波振幅指的是从基线（等电位线）到P波最高（或最低）峰值的垂直距离，通常在标准肢体导联（如II导联）上测量，本例中的P波振幅为{p_amp}mV。P波时限指的是P波从起点到终点的水平时间，反映心房除极的总时长，本例中的P波时限为{p_dur}ms。P波离散度指的是12导联心电图中最长P波时限与最短P波时限的差值，反映心房电活动的不均一性，本例中的P波离散度为{p_disp}ms。P波切迹指P波中出现一个或多个次峰或凹陷，通常表现为双峰或多峰形态，本例中{'有' if p_notch else '无'}P波切迹。"
                f"QRS波群反映左、右心室除极电位和时间的变化，第一个向下的波为Q波，向上的波为R波，接着向下的波是S波。R波振幅是指QRS复合波中向上的最高峰与基线之间的垂直距离，本例中的R波振幅为{r_amp}mV。S波深度指QRS复合波中向下的最低点与基线之间的深度，常用胸前导联测量，本例中的S波深度为{s_dep}mV。"
                f"病理性Q波是指心电图上Q波出现异常，表现为时限变宽，宽度大于0.04秒，Q波振幅增大，大于同一导联R波的1/4，本例中{'有' if path_q else '无'}病理性Q波。"
                f"碎裂QRS波是指常规12导联心电图上相邻两个或更多导联上出现各种形式的RSR'波形，有或无病理性Q波，除外束支传导阻滞，且QRS时限＜120 ms，包括QRS波群中出现≥1个R'波或R波有切迹，或S波有切迹，本例中{'有' if frag_q else '无'}碎裂QRS波。"
                f"R/S比值指的是同一导联中R波振幅与S波深度的比值，本例中V1导联的R/S比值为:{rs_v1}, V2导联的R/S比值为:{rs_v2},V3导联的R/S比值为:{rs_v3}, V4导联的R/S比值为:{rs_v4},V5导联的R/S比值为:{rs_v5}, V6导联的R/S比值为:{rs_v6}。"
                f"ST段是指由QRS波群结束到T波开始的平线，反映心室各部均在兴奋而各部处于去极化状态。心电图ST段改变，一般是指ST段发生了向上或向下的偏移，反映了心肌电活动情况，本例中的ST段偏移{st_dev}mV，ST段斜率指的是ST段与水平线的倾斜角度，反映心肌复极状态，本例中的ST段斜率{st_slope}mV/ms。"
                f"T波指的是继QRS波群后的一个波幅较低而波宽较长的电波，反映心室兴奋后再极化过程。T波振幅是指T波最高点（正向T波）或最低点（倒置T波）与基线之间的电位差，本例中的T波振幅为{t_amp}mV，T波对称性描述的是T波上升阶段与下降阶段在时间上的比值，本例中的T波对称性为{t_sym}。"
                f"U波振幅是指U波峰值与基线之间的幅值，本例中的U波振幅为{u_amp}mV。"
            )},
            {"role": "user",    "content": "请运用向量心电图原理，分析额面及水平面心电向量特征"},
            {"role": "assistant", "content": (
                f"额面电轴反映心脏电活动在额状面（冠状面）的总体方向，通常通过肢体导联（I、II、III、aVR、aVL、aVF）分析，额面电轴基于肢体导联提供心脏去极和复极在前额平面的投影，是判断心脏电活动主要方向的重要依据。QRS电轴指心脏除极和复极时额面最大综合向量与水平轴形成的角度，本例中的QRS电轴为{qrs_ax}°，T波电轴指的是额面T波的平均电轴，本例中的T波电轴为{t_ax}°，QRS-T夹角是指去极化方向和复极化方向之间围成的夹角,主要反映继发性心室复极改变，本例中的QRS-T夹角为{qrst_ang}°。"
                f"水平面电轴反映心脏电活动在横断面（水平面）的总体方向，通过胸导联（V1-V6）分析，反映心肌电活动在胸前导联投射下的分布特征，有助于局部区域电活动的研究和定位。R/S转换导联是指在胸前导联中，第一个出现R波与S波幅度相等或相近的导联，本例中的R/S转换导联为{rs_lead}，R波递增斜率描述的是胸前导联中R波上升部分的斜率，即R波从起点到峰值的电位变化速率，本例中的R波递增斜率为{r_slope}mV/ms。"
            )},
            {"role": "user",    "content": "请对这张心电图图像的结构化病变指数进行详细分析，包括Sokolow-Lyon指数、Cornell电压指数和Cornell乘积指数"},
            {"role": "assistant", "content": (
                f"Sokolow-Lyon指数是一种用于评估心脏电图上左心室肥厚程度的指数，本例中的Sokolow-Lyon指数为{sok}mV，"
                f"Cornell 电压指数和 Cornell 乘积指数是90 年代出现的新标准，其不仅对左室肥大的检出率和患者死亡率的预测价值高，而且计算方法简单，更适合临床应用。本例中的Cornell电压指数为{cornV}mV，Cornell乘积指数为{cornP}mV。"
            )},
            {"role": "user",    "content": "请给出结构化诊断报告"},
            {"role": "assistant", "content": (
                f"综合诊断结论：主要诊断：{concl.get('primary_diagnosis','') or '暂无'}，"
                f"次要诊断：{concl.get('secondary_diagnosis','') or '暂无'}，"
                f"补充诊断：{concl.get('supplementary_description','') or '暂无'}，"
                f"扩展诊断：{ext_ann}。"
            )}
        ]

        # images 字段对应 JSON 注释中的 data_paths/ecg_plot 字段
        img_rel = note.get('data_paths', {}).get('ecg_plot', '')
        images  = [os.path.join('mimic_data', img_rel)] if img_rel else []

        return {'messages': messages, 'images': images}

    except Exception as e:
        return {'error': record_id, 'message': str(e)}

if __name__ == '__main__':
    # 从 CSV 中读取待处理 JSON 路径
    json_paths = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get('json_path')
            if path and path.endswith('.json'):
                json_paths.append(path)

    # 并行处理
    num_workers = cpu_count()
    results = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_file, json_paths), total=len(json_paths)):
            results.append(res)

    # 分离成功结果与错误日志
    llama_data, errors = [], []
    for res in results:
        if 'error' in res:
            errors.append(f"{res['error']}: {res['message']}")
        else:
            llama_data.append(res)

    # 写入 LLama Factory 数据和错误日志
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llama_data, f, ensure_ascii=False, indent=2)
    if errors:
        with open(error_log, 'w', encoding='utf-8') as elog:
            elog.writelines(line + '\n' for line in errors)

    print(f"已生成 LLama Factory 数据，保存在：{output_path}")
    if errors:
        print(f"处理过程中发生错误，详情请见：{error_log}")
