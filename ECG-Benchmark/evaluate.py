from run_inference import run_qwenvl_inference
from eval_class import eval_class
from eval_report import eval_report
import argparse
import os

def validate_paths(args):
    """验证输入路径并创建输出目录"""
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.isdir(args.image_folder):
        raise NotADirectoryError(f"Image folder not found: {args.image_folder}")
    if not os.path.isfile(args.question_file):
        raise FileNotFoundError(f"Question file not found: {args.question_file}")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate ECG inference tasks.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the inference model directory or file")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Directory containing ECG images to process")
    parser.add_argument("--question-file", type=str, required=True,
                        help="JSON file with questions/prompts for evaluation")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="Output JSON file for storing model answers")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens for text generation (default: 1024)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["2_classes", "5_classes", "23_classes","new_2_classes","new_5_classes","9_classes","report"],
                        help="Evaluation mode: classification (e.g. 2_classes, 5_classes) or report")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--nproc", type=int, default=8,
                        help="Number of parallel LLM workers (default: 8)")
    args = parser.parse_args()

    # 验证路径并创建必要目录
    try:
        args = validate_paths(args)
    except Exception as e:
        print(f"Path validation error: {e}")
        exit(1)

    # 运行推理
    try:
        run_qwenvl_inference(args)
        print(f"Inference completed. Answers saved to {args.answers_file}")
    except Exception as e:
        print(f"Inference error: {e}")
        exit(1)

    # 运行评估
    try:
        if args.mode == "report":
            eval_report(args)
        else:
            eval_class(args)
        print("Evaluation completed successfully.")
    except Exception as e:
        print(f"Evaluation error: {e}")
        exit(1)
