import os
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from torch.nn import CrossEntropyLoss
from multiprocessing import Process, Queue, current_process

PROMPT_DICT = {
    'alpaca': {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task.\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    },
    'wiz': {
        "prompt_input": "{instruction}\n{input}\n\n### Response:",
        "prompt_no_input": "{instruction}\n\n### Response:"
    },
    'vicuna': {
        "prompt_input": (
            "A chat between a user and assistant. USER: {instruction}\nInput:\nAn image provided ASSISTANT:"
        ),
        "prompt_no_input": (
            "A chat between a user and assistant. USER: {instruction} ASSISTANT:"
        )
    }
}

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("JanusEvaluator")

class JanusEvaluator:
    def __init__(self, model_path, device_id):
        self.device = f"cuda:{device_id}"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(torch.bfloat16).to(self.device).eval()
        self.logger = setup_logger()

    def _generate_logits(self, text, image):
        conversation = [
            {"role": "<|User|>", "content": text, "images": [image]},
            {"role": "<|Assistant|>", "content": ""}
        ]
        pil_images = load_pil_images(conversation)
        inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(self.device)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**inputs)
        with torch.no_grad():
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        return torch.cat(outputs.scores, dim=0)

    def _compute_loss(self, logits, target_text, partial=False):
        target_ids = self.tokenizer(target_text, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(self.device)
        if partial:
            start_token = len(self.tokenizer.encode(target_text))
            target_ids[0, :target_ids.size(1) - start_token] = -100
        logits = torch.where(torch.isinf(logits), torch.tensor(0.0, device=logits.device), logits)
        if logits.size(0) < target_ids.size(1):
            pad = logits.new_zeros((target_ids.size(1) - logits.size(0), logits.size(1)))
            logits = torch.cat([logits, pad], dim=0)
        elif logits.size(0) > target_ids.size(1):
            pad = target_ids.new_zeros((1, logits.size(0) - target_ids.size(1)))
            target_ids = torch.cat([target_ids, pad], dim=1)
        loss_fn = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fn(logits, target_ids.view(-1))
        return torch.exp(loss).item(), loss.item()

    def process_one(self, sample, prompt_type):
        try:
            sample_id = sample.get("id", None)
            instruction = sample["instruction"]
            output = sample["output"]
            image_path = sample.get("input", None)
            if not image_path:
                return None

            prompt_set = PROMPT_DICT[prompt_type]
            prompt_full = prompt_set["prompt_input"].format_map({"instruction": instruction, "input": image_path})
            reverse_prompt = "Below is the response..." + output + "\nGenerate the instruction for the above response."
            reverse_prompt_full = prompt_set["prompt_no_input"].format_map({"instruction": reverse_prompt}) + instruction

            logits_ins = self._generate_logits(instruction, image_path)
            logits_out = self._generate_logits(output, image_path)
            logits_ins_cond = self._generate_logits(reverse_prompt_full, image_path)
            logits_out_cond = self._generate_logits(prompt_full + output, image_path)

            ppl_ins, loss_ins = self._compute_loss(logits_ins, instruction)
            ppl_out, loss_out = self._compute_loss(logits_out, output)
            ppl_ins_cond, loss_ins_cond = self._compute_loss(logits_ins_cond, instruction, partial=True)
            ppl_out_cond, loss_out_cond = self._compute_loss(logits_out_cond, output, partial=True)

            return {
                "id": sample_id,
                "ppl": [ppl_ins, ppl_out, ppl_ins_cond, ppl_out_cond],
                "loss": [loss_ins, loss_out, loss_ins_cond, loss_out_cond]
            }

        except Exception as e:
            self.logger.warning(f"Failed processing sample {sample.get('id', None)}: {e}")
            return None

def worker(rank, data_chunk, model_path, output_queue, prompt_type):
    evaluator = JanusEvaluator(model_path, rank)
    results = []
    for i, sample in enumerate(tqdm(data_chunk, desc=f"GPU {rank}", position=rank)):
        result = evaluator.process_one(sample, prompt_type)
        if result:
            results.append(result)
            # 阶段性保存
            if (i + 1) % 100 == 0:
                output_queue.put((rank, results))
                results = []
    # 最后一批结果
    if results:
        output_queue.put((rank, results))
    output_queue.put((rank, "DONE"))

def run_multigpu_eval(model_path, data_path, save_path, prompt_type="vicuna"):
    with open(data_path, "r") as f:
        data = json.load(f)

    num_gpus = torch.cuda.device_count()
    split_size = len(data) // num_gpus
    splits = [data[i * split_size:(i + 1) * split_size] for i in range(num_gpus)]
    if len(data) % num_gpus:
        splits[-1].extend(data[num_gpus * split_size:])

    output_queue = Queue()
    processes = []
    for rank in range(num_gpus):
        p = Process(target=worker, args=(rank, splits[rank], model_path, output_queue, prompt_type))
        p.start()
        processes.append(p)

    merged = [[] for _ in range(num_gpus)]
    finished = 0
    while finished < num_gpus:
        rank, result_batch = output_queue.get()
        if result_batch == "DONE":
            finished += 1
        else:
            merged[rank].extend(result_batch)
            with open(f"{save_path}_gpu{rank}.jsonl", "a") as f:
                for item in result_batch:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for p in processes:
        p.join()

    # 合并所有结果
    all_results = []
    for rank in range(num_gpus):
        temp_path = f"{save_path}_gpu{rank}.jsonl"
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                all_results.extend([json.loads(line) for line in f])
            os.remove(temp_path)
    with open(save_path, "w") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
