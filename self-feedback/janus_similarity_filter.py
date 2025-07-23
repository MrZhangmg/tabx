import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from multiprocessing import Process, Queue, current_process
from tqdm import tqdm


def mean_pool_last_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_hidden = hidden_states * mask
    summed = torch.sum(masked_hidden, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / summed_mask


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b).item()


def get_embedding_from_output(model, tokenizer, processor, instruction: str, output_text: str, image_path: str, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{instruction}", "images": ["image_placeholder"]},
        {"role": "<|Assistant|>", "content": output_text}
    ]
    inputs = processor(conversations=conversation, images=[image], force_batchify=True).to(device)
    if hasattr(inputs, "pixel_values"):
        inputs.pixel_values = inputs.pixel_values.to(torch.float32)
    with torch.no_grad():
        outputs = model.language_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = mean_pool_last_hidden(last_hidden, inputs.attention_mask)
    return pooled


def generate_output(model, tokenizer, processor, instruction: str, image_path: str, device: str) -> str:
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{instruction}", "images": ["image_placeholder"]},
        {"role": "<|Assistant|>", "content": ""}
    ]
    inputs = processor(conversations=conversation, images=[image], force_batchify=True).to(device)
    if hasattr(inputs, "pixel_values"):
        inputs.pixel_values = inputs.pixel_values.to(torch.float32)
    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    with torch.no_grad():
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            do_sample=False,
            return_dict_in_generate=False
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def worker(data_chunk, model_path, threshold, output_queue, device_id, retained_path, filtered_path):
    device = f"cuda:{device_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MultiModalityCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
    processor = VLChatProcessor.from_pretrained(model_path)

    retained, filtered = [], []
    for idx, item in enumerate(tqdm(data_chunk, desc=f"Process-{device_id}", position=device_id)):
        sample_id = item.get("id", None)
        if not sample_id:
            continue
        try:
            instruction = item["instruction"]
            output = item["output"]
            image_path = item["input"]
            output_embedding = get_embedding_from_output(model, tokenizer, processor, instruction, output, image_path, device)
            generated_output = generate_output(model, tokenizer, processor, instruction, image_path, device)
            generated_embedding = get_embedding_from_output(model, tokenizer, processor, instruction, generated_output, image_path, device)
            sim = cosine_similarity(output_embedding, generated_embedding)
            if sim >= threshold:
                retained.append(item)
            else:
                filtered.append(item)
        except Exception as e:
            print(f"[{current_process().name}] 跳过样本 {sample_id}, 出错: {repr(e)}")

        # 每50条写入一次
        if (idx + 1) % 50 == 0:
            if retained:
                with open(retained_path, 'a', encoding='utf-8') as f_ret:
                    for r in retained:
                        f_ret.write(json.dumps(r, ensure_ascii=False) + "\n")
                retained.clear()
            if filtered:
                with open(filtered_path, 'a', encoding='utf-8') as f_filt:
                    for r in filtered:
                        f_filt.write(json.dumps(r, ensure_ascii=False) + "\n")
                filtered.clear()

    # 剩余未写入的
    output_queue.put((retained, filtered))


def janus_similarity_filter_multi_gpu(
    final_merged_output_path: str,
    output_save_path: str,
    filtered_save_path: str,
    threshold: float,
    model_name_or_path: str,
    num_gpus: int = 4
) -> List[Dict[str, Any]]:
    import os
    from multiprocessing import Process, Queue
    import json

    with open(final_merged_output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_size = (len(data) + num_gpus - 1) // num_gpus
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    output_queue = Queue()
    processes = []
    temp_retained_files = []
    temp_filtered_files = []

    for i in range(num_gpus):
        retained_path = f"{output_save_path}_gpu{i}.jsonl"
        filtered_path = f"{filtered_save_path}_gpu{i}.jsonl"
        temp_retained_files.append(retained_path)
        temp_filtered_files.append(filtered_path)
        # 确保路径存在
        os.makedirs(os.path.dirname(retained_path), exist_ok=True)
        os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
        p = Process(target=worker, args=(chunks[i], model_name_or_path, threshold, output_queue, i, retained_path, filtered_path))
        p.start()
        processes.append(p)

    all_retained, all_filtered = [], []

    # 写入每个进程剩余的 retained 和 filtered（未达到50条的部分）
    for i in range(num_gpus):
        retained, filtered = output_queue.get()
        all_retained.extend(retained)
        all_filtered.extend(filtered)

        # 把剩余的也写入临时文件中
        with open(temp_retained_files[i], 'a', encoding='utf-8') as f_ret:
            for r in retained:
                f_ret.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(temp_filtered_files[i], 'a', encoding='utf-8') as f_filt:
            for r in filtered:
                f_filt.write(json.dumps(r, ensure_ascii=False) + "\n")

    for p in processes:
        p.join()

    # 合并所有 retained
    with open(output_save_path, 'w', encoding='utf-8') as f:
        for fname in temp_retained_files:
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as part:
                    for line in part:
                        f.write(line)
                os.remove(fname)

    # 合并所有 filtered
    with open(filtered_save_path, 'w', encoding='utf-8') as f:
        for fname in temp_filtered_files:
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as part:
                    for line in part:
                        f.write(line)
                os.remove(fname)

    return all_filtered

