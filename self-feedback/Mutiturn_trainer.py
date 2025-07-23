import os
import shutil
import asyncio
from janus_similarity_filter import janus_similarity_filter_multi_gpu
from teacher_augumentation import dispatch_dashscope_concurrent
from janus_ppl_evaluator import run_multigpu_eval
from ifd_selector_and_merge import ifd_based_decision_and_merge
from finetune_train import train_janus_model
import torch.multiprocessing as mp

def main():
    model_path = "./Janus-Pro-7B"
    merged_data = "./data/recycle_finetune/merge/xxx.json"

    for i in range(3): 

    # 0. 保存模型快照
        backup_model_path = f"{model_path}_round{i+1}_backup"
        if os.path.exists(model_path):
            if os.path.exists(backup_model_path):
                shutil.rmtree(backup_model_path)
            shutil.copytree(model_path, backup_model_path)
            print(f"model checkpoint has been saved to {backup_model_path}")

    # 1. 相似度筛选
        retained_path = f"./data/recycle_finetune/retained_data/retained_round{i+2}.jsonl"
        filtered_path = f"./data/recycle_finetune/filter_data/filter_round{i+2}.jsonl"
        filtered_samples = janus_similarity_filter_multi_gpu(
            final_merged_output_path=merged_data,
            output_save_path=retained_path,
            filtered_save_path=filtered_path,
            threshold=0.9,
            model_name_or_path=model_path,
            num_gpus=4
        )

    # 2. 教师增强（生成新的样本）
        new_aug_output = f"./data/recycle_finetune/augumentation/aug_outputs_round{i+2}.json"
        old_data_output = f"./data/recycle_finetune/old_data/old_data_round{i+2}.json"
      # 用上一步返回的列表作为输入

        asyncio.run(dispatch_dashscope_concurrent(
            data_path=filtered_path,
            image_root="./",
            model="qwen-vl-max",
            save_path=new_aug_output,
            concurrency=20,
            save_every=20,
            max_tokens=512,
            old_data_path=old_data_output
        ))

    # 3. Janus-PPL 评估
        metrics1_path = f"./data/recycle_finetune/metric/metrics1_round{i+2}.jsonl"
        metrics2_path = f"./data/recycle_finetune/metric/metrics2_round{i+2}.jsonl"

        run_multigpu_eval(
            model_path=model_path, 
            data_path=new_aug_output,
            save_path=metrics1_path,
            prompt_type="vicuna",
         )

        run_multigpu_eval(
            model_path=model_path,
            data_path=old_data_output,
            save_path=metrics2_path,
            prompt_type="vicuna",
        )

    # 4. 选择并合并
        decision_path = f"./data/recycle_finetune/decision/decision_round{i+2}.jsonl"
        merged_data = f"./data/recycle_finetune/merge/merged_output_round{i+2}.json"

        ifd_based_decision_and_merge(
            metrics_file1_path=metrics1_path,
            metrics_file2_path=metrics2_path,
            raw_file1_path=new_aug_output,
            raw_file2_path=old_data_output,
            retained_path=retained_path,
            decision_output_path=decision_path,
            final_merged_output_path=merged_data,
        )

    # 5. 微调 Janus-Pro 模型
        train_janus_model(
            data_dir=merged_data,
            pretrained_model_path=model_path,
            output_dir=model_path  # 每轮覆盖原始模型
        )
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
