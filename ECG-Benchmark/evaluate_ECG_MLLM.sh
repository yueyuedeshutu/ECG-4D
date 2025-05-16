ptb-xl 的评估代码
python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
    --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
    --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/bench_2_class.json"\
    --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/ptb_xl_2_class_output-ecgmllm-1000-v2.json"\
    --mode "2_classes"

python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
    --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
    --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/bench_5_class.json"\
    --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/ptb_xl_5_class_output-ecgmllm-1000-v2.json"\
    --mode "5_classes"

python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
    --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc-1000"\
    --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
    --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/bench_23_class.json"\
    --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/ptb_xl_23_class_output-ecgmllm-1000-v2.json"\
    --mode "23_classes"

python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
    --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
    --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
    --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/update_bench_2_class.json"\
    --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/ptb_xl_update_2_class_output-ecgmllm.json"\
    --mode "new_2_classes"

python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
    --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
    --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
    --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/update_bench_5_class.json"\
    --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/ptb_xl_updete_5_class_output-ecgmllm.json"\
    --mode "new_5_classes"

# python /gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/evaluation/evaluate.py\
#     --model-path "/gpfsdata/home/liuxinyue/ECG/LLaMA-Factory/models_result/qwen2.5_vl_7B_lora_sft_class_cpsc"\
#     --image-folder "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/image"\
#     --question-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/bench/bench_report_ceshi.json"\
#     --answers-file "/gpfsdata/home/liuxinyue/ECG/ECG_Benchmark_final/PTB-XL/result/mllm/ptb_xl_report_output-ecgmllm-v1.json"\
#     --mode "report"
