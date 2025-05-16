import os
import json
import random
import re
random.seed(42)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score,hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_f1_auc(y_pred, y_true):
    #计算多标签分类任务中的 平均 F1 分数、平均 AUC 分数 和 Hamming Loss（汉明损失）。
    # Binarize labels
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    # print(y_true)
    # print(y_true_bin)
    hl = hamming_loss(y_true_bin, y_pred_bin)
    
    f1_scores_all = []
    # Compute the F1 score
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=None)
    for idx, cls in enumerate(mlb.classes_):
        # print(f'F1 score for class {cls}: {f1_scores[idx]}')
        f1_scores_all.append(f1_scores[idx])
    
    # Compute the AUC score
    auc_scores = []
    for i in range(y_true_bin.shape[1]):
        try:
            auc = roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i])
        except ValueError:
            auc = np.nan  #If AUC cannot be calculated, NaN is returned
        auc_scores.append(auc)
        # print(f'AUC score for class {mlb.classes_[i]}: {auc}')    
    # print("f1 all",np.mean(f1_scores_all), "auc all", np.mean(auc_scores))
    return np.mean(f1_scores_all), np.mean(auc_scores), hl

def eval_class(args):
    #所有可能的标签
    if args.mode in ["2_classes", "new_2_classes"]:
        label_space = ["NORMAL","ABNORMAL"]
    elif args.mode == "5_classes":
        label_space=["NORM","CD","MI","STTC","HYP"]
    elif args.mode == "new_5_classes":
        valid_options = ["A", "B", "C", "D", "E"]
    elif args.mode=="9_classes":
        label_space=[
            "NORM",  # Normal ECG
            "AF",    # Atrial fibrillation
            "I-AVB", # First-degree atrioventricular block
            "LBBB",  # Left bundle branch block
            "RBBB",  # Right bundle branch block
            "PAC",   # Premature atrial contraction
            "PVC",   # Premature ventricular contraction
            "STD",   # ST-segment depression
            "STE"    # ST-segment elevated
        ]
    elif args.mode=="23_classes":
        label_space=[
            "NORM",  # Normal ECG
            "STTC",  # ST/T Change
            "NST_",  # Nonspecific ST Changes
            "ISC_",  # ischemic ST-T changes
            "ISCA",  # Ischemic in Anterior Leads
            "ISCI",  # Ischemic in Inferior Leads
            "IMI",   # Inferior Myocardial Infarction
            "AMI",   # Anterior Myocardial Infarction
            "PMI",   # Posterior Myocardial Infarction
            "LMI",   # Lateral Myocardial Infarction
            "LVH",   # Left Ventricular Hypertrophy
            "RVH",   # Right Ventricular Hypertrophy
            "LAO/LAE",  # Left Atrial Overload/Enlargement
            "RAO/RAE",  # Right Atrial Overload/Enlargement
            "LAFB/LPFB",  # Left Anterior/Posterior Fascicular Block
            "CLBBB",  # Complete Left Bundle Branch Block
            "CRBBB",  # Complete Right Bundle Branch Block
            "IRBBB",  # Incomplete Right Bundle Branch Block
            "ILBBB",  # Incomplete Left Bundle Branch Block
            "SEHYP",  # Septal Hypertrophy
            "IVCD",   # Nonspecific Intraventricular Conduction Disturbance
            "WPW",    # Wolff-Parkinson-White Syndrome
            "_AVB"    # AV Block
        ]
    else:
        print("error args.mode:",args.mode)
    file=args.answers_file
    score_dict = {}
    predict_list = []
    golden_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 提取预测标签（处理text或response字段）
            pred_text = item.get("text", item.get("response", "")).strip().upper()
            # 提取真实标签（处理answer或ground_truth字段）
            gt_text = item.get("answer", item.get("ground_truth", "")).strip().upper()

            # 跳过任意一个为空的情况
            if not pred_text or not gt_text:
                continue

            if args.mode == "2_classes":
                # 2分类模式：直接匹配整个字符串
                predict = [label for label in label_space if label == pred_text]
            elif args.mode =="new_2_classes":
                #单选题
                # 从 text 中提取选项标记（如 "A" 或 "B"）
                pred_mark = pred_text.split('.')[0]  # 提取 "A" 或 "B"
                # 转换为 label_space 中的标签
                predict = ["NORMAL"] if pred_mark == "A" else ["ABNORMAL"]
            elif args.mode == "new_5_classes":
                # 多选题：支持 "Answer: B", "Answer: B;C;D", "Answer: C.MI(...)", "Answer: B.CD(...);C.MI(...)"
                predict = re.findall(r'\b([A-E])\b(?=[\.\);]|$)', pred_text)
                predict=[i for i in predict if i in valid_options]
            else:
                # 多分类模式：按分号分割并验证每个标签
                pred_labels = [s.strip() for s in pred_text.split(";") if s.strip()]
                predict = [label for label in pred_labels if label in label_space]
            

            if args.mode == "2_classes":
                # 2分类模式：直接匹配整个字符串
                answer = [label for label in label_space if label == gt_text]
            elif args.mode =="new_2_classes":
                #单选题
                # 使用正则表达式提取 "Answer: " 后的标记
                gt_match = re.match(r"ANSWER: (\w)\.", gt_text)
                if not gt_match:
                    raise ValueError(f"Invalid ground_truth format: {gt_text}")
                gt_mark = gt_match.group(1)  # 提取 "A" 或 "B"
                # 转换为 label_space 中的标签
                answer = ["NORMAL"] if gt_mark == "A" else ["ABNORMAL"]
            elif args.mode == "new_5_classes":
                # 多选题：提取多个标签（如 "Answer: B.CD(...);C.MI(...);"）
                answer = re.findall(r'\b([A-E])\b(?=[\.\);]|$)', gt_text)
                answer=[i for i in answer if i in valid_options]
            else:
                # 多分类模式：按分号分割并验证每个标签
                gt_labels = [s.strip() for s in gt_text.split(";") if s.strip()]
                answer = [label for label in gt_labels if label in label_space]

            predict_list.append(predict)
            golden_list.append(answer)
        print("Predictions:", predict_list[:10])
        print("Ground truth:", golden_list[:10])
        f1, auc, hl = compute_f1_auc(predict_list, golden_list)
        print(file, "f1", round(f1*100, 1), "auc", round(auc*100, 1), "hl", round(hl*100, 1))
