from multimodal_trainer import EnhancedMultiModalTrainer
from typing import Optional, Dict

def train_janus_model(
    data_dir: str,
    pretrained_model_path: str,
    output_dir: str,
    batch_size: int = 2,
    max_epochs: int = 2,
    lr: float = 2e-5,
    user_question: str = "",
    optimizer_name: str = "AdamW",
    lora_config: Optional[Dict] = None,
    training_args: Optional[Dict] = None
):

    lora_config = lora_config or {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    training_args = training_args or {
        "fp16": True,
        "max_grad_norm": 1.0,
        "save_strategy": "epoch",
        "evaluation_strategy": "no",
        "gradient_accumulation_steps": 4,
        "logging_steps": 50,
        "save_total_limit": 2,
        "remove_unused_columns": False,
    }

    trainer = EnhancedMultiModalTrainer(
        data_dir=data_dir,
        pretrained_model_path=pretrained_model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr=lr,
        user_question=user_question,
        optimizer_name=optimizer_name,
        lora_config=lora_config,
        training_args=training_args,
    )

    trainer.train()
