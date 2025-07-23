# multimodal_cli.py
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
from io import BytesIO
import argparse

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True
    )
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16).cuda()
    else:
        model = model.to(torch.float16)
    return model

def main():
    parser = argparse.ArgumentParser(description="Multimodal Chat Model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    # 初始化模型和处理器
    model_path = "./Janus-Pro-7B"
    model = load_model(model_path)
    processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    # 加载图片
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 构建对话
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{args.question}",
            "images": ["image_placeholder"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 处理输入并生成回答
    inputs = processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    inputs_embeds = model.prepare_inputs_embeds(**inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
