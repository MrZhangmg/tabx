# multimodal_trainer.py
import gc
import os
from typing import List, Tuple, Dict, Any
import torch
import json
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW, SGD
import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments, get_scheduler
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from transformers import TrainerCallback
from torch.cuda import OutOfMemoryError
from builtins import Exception

class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

class EnhancedMultiModalModel(MultiModalityCausalLM):
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_token_masks=None,
        labels=None,
        **kwargs,
    ):
        
        if pixel_values is not None and image_token_masks is not None:
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        kwargs.pop("inputs_embeds", None)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _process_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_token_masks: torch.BoolTensor,
    ) -> torch.Tensor:
        bs, n = pixel_values.shape[0:2]
        images = pixel_values.view(bs * n, *pixel_values.shape[2:])
        image_features = self.vision_model(images)
        aligned_features = self.aligner(image_features)

        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:])
        aligned_features = aligned_features.flatten(1, 2)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        for i in range(bs):
            num_image_tokens = image_token_masks[i].sum().item()
            num_aligned_tokens = aligned_features[i].shape[0]
            if num_image_tokens != num_aligned_tokens:
                raise ValueError(
                    f"Mismatch at sample {i}: "
                    f"image_token_masks has {num_image_tokens} tokens, "
                    f"but aligned_features has {num_aligned_tokens} tokens!"
                )
            text_embeds[i][image_token_masks[i]] = aligned_features[i]

        return text_embeds


class EnhancedMultiModalTrainer:
    def __init__(self, 
                 data_dir: str, 
                 pretrained_model_path: str, 
                 output_dir: str, 
                 batch_size: int = 1, 
                 max_epochs: int = 10, 
                 lr: float = 3e-4, 
                 user_question: str = "",
                 optimizer_name: str = "AdamW",
                 lora_config: dict = None,
                 training_args: dict = None):
        
        self.data_dir = data_dir
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.user_question = user_question
        self.optimizer_name = optimizer_name
        self.lora_config = lora_config if lora_config else {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.training_args = training_args if training_args else {
            "fp16": True,
            "max_grad_norm": 1.0,
            "save_strategy": "epoch",
            "evaluation_strategy": "no",
            "logging_steps": 50,
            "save_total_limit": 2,
            "remove_unused_columns": False,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self) -> List[Tuple[str, str]]:
        """加载图像和文本配对数据。"""
        pairs = []
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        json_path = self.data_dir
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
        # 检查 JSON 对象是否包含所需的字段
            if "input" not in item or "output" not in item or "instruction" not in item:
                raise ValueError(f"Invalid JSON structure: missing 'image', 'output', or 'user_question' field in {item}")
            
            image_path = item["input"]
            text_content = item["output"]
            user_question = item["instruction"]

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            pairs.append({
                "image_path": image_path,
                "text": text_content,
                "user_question": user_question,
            })
        if not pairs:
            raise ValueError(f"No valid image-text pairs found in {json_path}.")
        
        return pairs

    def _prepare_model(self):
        """加载预训练模型并配置 LoRA 微调。"""
        print(f"Loading pretrained model from {self.pretrained_model_path}")
        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained_model_path,
            slow_image_processor_class="AutoImageProcessor"
        )
        self.model = EnhancedMultiModalModel.from_pretrained(
            self.pretrained_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 配置 LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params} / {total_params}")

    def _prepare_optimizer_and_scheduler(self, dataset_size: int):
        """准备优化器和学习率调度器。"""
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_name == "8bit-AdamW":
            optimizer = bnb.optim.AdamW8bit(self.model.parameters(),lr=self.lr,betas=(0.9, 0.999),weight_decay=0.01)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        num_training_steps = dataset_size * self.max_epochs // self.batch_size
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler_name = self.training_args.get("lr_scheduler_type", "cosine")
        if scheduler_name not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """自定义数据整理函数。"""
        conversations = []
        for item in batch:
            conversations.extend(self._generate_conversation(
                image_path=item["image_path"], 
                assistant_text=item["text"],
                user_question=item["user_question"],
                ))
        
        pil_images_list = load_pil_images(conversations)
        encoded = self.processor(
            conversations=conversations,
            images=pil_images_list,
            return_tensors="pt",
            max_token_new=512,
            force_batchify=True,
        )
        encoded["labels"] = encoded["input_ids"].clone()
        image_placeholder_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image_placeholder>")
        input_ids = encoded["input_ids"]
        image_token_masks = (input_ids == image_placeholder_token_id)
        
        if not image_token_masks.any():
            raise ValueError("No <image_placeholder> tokens found in the input!")

        encoded["image_token_masks"] = image_token_masks
        return dict(encoded)

    def _generate_conversation(self, image_path: str, assistant_text: str, user_question: str) -> List[Dict[str, Any]]:
        """生成对话模板。"""
        eos_token = self.processor.tokenizer.eos_token 
        if not assistant_text.endswith(eos_token): 
            assistant_text = assistant_text + eos_token

        return [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{user_question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": assistant_text},
        ]

    

    
        
    def train(self):
        """主训练流程。"""
        try:
            pairs = self._load_data()
            dataset = pairs
            self._prepare_model()
            self._prepare_optimizer_and_scheduler(len(dataset))

            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=self.max_epochs,
                per_device_train_batch_size=self.batch_size,
                learning_rate=self.lr,
                **self.training_args,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=self._collate_fn,
                optimizers=(self.optimizer, self.lr_scheduler),
                callbacks=[MemoryMonitorCallback()],
            )

            print("[Start!] Start training!!!!!---------->>>>>>>")
            trainer.train()

            print("[on Progress] Merging LoRA weights...")
            self.model = self.model.merge_and_unload()

            print("[on Progress] Saving...")
            self.model.save_pretrained(self.output_dir)
            self.processor.save_pretrained(self.output_dir)
            print(f"[Done!] Fine-tuned model have saved to {self.output_dir}")
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            emergency_save_dir = os.path.join(self.output_dir, "emergency_save_dir")
            os.makedirs(emergency_save_dir, exist_ok=True)
            self.model.to("cpu")
            self.model.save_pretrained(emergency_save_dir)
            self.processor.save_pretrained(emergency_save_dir)
            print(f"[Emergency Save] Model saved to {emergency_save_dir}")
            raise e
