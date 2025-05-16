ptb-xl 的评估代码
python ./evaluation/evaluate.py\
    --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "./image"\
    --question-file "./bench/bench_2_class.json"\
    --answers-file "./result/ptb_xl_2_class_output-ecgmllm-1000-v2.json"\
    --mode "2_classes"

python ./evaluation/evaluate.py\
    --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "./image"\
    --question-file "./bench/bench_5_class.json"\
    --answers-file "./result/ptb_xl_5_class_output-ecgmllm-1000-v2.json"\
    --mode "5_classes"

python ./evaluation/evaluate.py\
    --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "./image"\
    --question-file "./bench/bench_23_class.json"\
    --answers-file "./result/ptb_xl_23_class_output-ecgmllm-1000-v2.json"\
    --mode "23_classes"

python ./evaluation/evaluate.py\
    --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
    --image-folder "./image"\
    --question-file "./bench/update_bench_2_class.json"\
    --answers-file "./result/ptb_xl_update_2_class_output-ecgmllm.json"\
    --mode "new_2_classes"

python ./evaluation/evaluate.py\
    --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
    --image-folder "./image"\
    --question-file "./bench/update_bench_5_class.json"\
    --answers-file "./result/ptb_xl_updete_5_class_output-ecgmllm.json"\
    --mode "new_5_classes"

# python ./evaluation/evaluate.py\
#     --model-path "./LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
#     --image-folder "./image"\
#     --question-file "./bench/bench_report_ceshi.json"\
#     --answers-file "./result/mllm/ptb_xl_report_output-ecgmllm-v1.json"\
#     --mode "report"
