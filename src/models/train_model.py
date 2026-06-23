import torch
import logging
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def initialize_and_train_model(train_dataset, val_dataset):
    model_checkpoint = "distilbert-base-uncased"
    num_labels = 2
    
    logging.info(f"Loading {model_checkpoint} model architecture to device...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("CRITICAL WARNING: Training on CPU. This will take hours.")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    logging.info("Initiating model training loop...")
    trainer.train()
    
    final_output_path = "./models/production_model"
    trainer.save_model(final_output_path)
    logging.info(f"Production model safely exported to {final_output_path}")
    return model
