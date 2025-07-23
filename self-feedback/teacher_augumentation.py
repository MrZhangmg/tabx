import json
import os
import time
import base64
import asyncio
import logging
import re
from typing import List, Dict, Any
import dashscope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_payload(messages, image_path):
    content = []
    for message in messages:
        content.append({"text": message["content"]})

    if image_path:
        base64_image = encode_image(image_path)
        content.insert(0, {
            "image": f"data:image/jpg;base64,{base64_image}"
        })

    return content

async def fetch(semaphore, item, model, max_tokens):
    async with semaphore:
        try:
            payload = {
                "api_key": "" ,#api key
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"text": "You are a helpful assistant that can analyze images and tables to answer questions."}]
                    },
                    {
                        "role": "user",
                        "content": build_payload(item["messages"], item.get("image"))
                    }
                ],
                "max_tokens": max_tokens,
                "vl_high_resolution_images": True
            }

            response = await asyncio.to_thread(dashscope.MultiModalConversation.call, **payload)
            content = response.output.choices[0].message.content[0]["text"]

            new_instruction_match = re.search(r"\[New Instruction\](.*?)\[End\]", content, re.DOTALL)
            new_answer_match = re.search(r"\[New Answer\](.*?)\[End\]", content, re.DOTALL)

            result = {
                "id": item["id"],
                "input": item.get("input", "")
            }

            if new_instruction_match:
                result["instruction"] = new_instruction_match.group(1).strip()
                result["output"] = item.get("output", "")
                result["metric"] = "IFD"
            elif new_answer_match:
                result["instruction"] = item.get("instruction", "")
                result["output"] = new_answer_match.group(1).strip()
                result["metric"] = "r-IFD"
            else:
                result["instruction"] = item.get("instruction", "")
                result["output"] = item.get("output", "")
                result["metric"] = "IFD"

            return result

        except Exception as e:
            logger.error(f"DashScope API request failed for id={item['id']}: {e}")
            return {"id": item["id"], "error": str(e)}

async def dispatch_dashscope_concurrent(
    data_path: str,
    image_root: str,
    model: str,
    save_path: str,
    old_data_path: str,
    concurrency: int = 20,
    save_every: int = 10,
    max_tokens: int = 512
):
    # 从 JSON 文件中加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    old_data_dict = {item["id"]: item.copy() for item in data}
    semaphore = asyncio.Semaphore(concurrency)
    buffer = []
    total_saved = 0
    results = []
    message_id_pairs = []

    for item in data:
        ins = item.get("instruction", "")
        outp = item.get("output", "")
        input_path = item.get("input", "")
        image_path = os.path.join(image_root, input_path) if image_root else input_path

        sys_prompt, user_prompt = gen_prompt_multimodal(ins, outp)
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        message_id_pairs.append({
            "id": item["id"],
            "messages": message,
            "image": image_path,
            "instruction": ins,
            "output": outp,
            "input": input_path
        })

    for i in range(0, len(message_id_pairs), concurrency):
        batch = message_id_pairs[i:i+concurrency]
        tasks = [fetch(semaphore, item, model, max_tokens) for item in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        buffer.extend(batch_results)

        for r in batch_results:
            if r.get("metric") and r["id"] in old_data_dict:
                old_data_dict[r["id"]]["metric"] = r["metric"]

        if len(buffer) >= save_every or i + concurrency >= len(message_id_pairs):
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []
            existing.extend(buffer)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            total_saved += len(buffer)
            logger.info(f"Saved {len(buffer)} results (Total: {total_saved})")
            buffer = []

    with open(old_data_path, "w", encoding="utf-8") as f:
        json.dump(list(old_data_dict.values()), f, ensure_ascii=False, indent=2)

    return results


def gen_prompt_multimodal(ins: str, outp: str) -> (str, str):
    sys_prompt = "You are a helpful assistant that can analyze images and tables to answer questions."
    prompt_template = (
        "There is an instruction and its answer:\nInstruction:\n{ins}\n\nAnswer:\n{outp}\n\n"
        "Determine whether the answer is correct given the instruction and image.\n"
        "If the answer is correct, then please rewrite the instruction using a more diverse expression while ensuring that the new instruction still corresponds to the original answer.Format your output using the structure:[New Instruction]new instruction[End].\n"
        "But If the answer is incorrect, Format your output using the structure: [New Answer] how to get the correct answer and the correct answer [End]."
    )
    user_prompt = prompt_template.format(ins=ins, outp=outp)
    return sys_prompt, user_prompt
