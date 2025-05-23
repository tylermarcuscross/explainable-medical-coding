import json
import logging
import random
import string
from pathlib import Path
import os

import hydra
import numpy as np
import torch
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Load environment variables
load_dotenv(find_dotenv())

# Configure PyTorch Dynamo
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.disable = True

TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="config",
)
def main(cfg: OmegaConf) -> None:
    if hasattr(cfg, "deterministic") and cfg.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    test_mode = cfg.get("test_mode", False)
    test_samples = cfg.get("test_samples", 100) if test_mode else None

    if test_mode:
        LOGGER.info(f"RUNNING IN TEST MODE with {test_samples} samples per split")

    if hasattr(cfg, "seed"):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    diagnosis_column = (
        cfg.data.diagnosis_column
        if hasattr(cfg.data, "diagnosis_column")
        else "diagnosis_codes"
    )
    procedure_column = (
        cfg.data.procedure_column
        if hasattr(cfg.data, "procedure_column")
        else "procedure_codes"
    )

    # Set target column from settings or config if provided
    target_col = (
        cfg.data.target_column if hasattr(cfg.data, "target_column") else TARGET_COLUMN
    )

    LOGGER.info(f"Using text column: {TEXT_COLUMN}")
    LOGGER.info(f"Using target column: {target_col}")
    LOGGER.info(f"Using diagnosis column: {diagnosis_column}")
    LOGGER.info(f"Using procedure column: {procedure_column}")

    # Load dataset - use streaming for initial processing if dataset is large
    dataset_path = Path(cfg.data.dataset_path)
    LOGGER.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(str(dataset_path))

    # If in test mode, take only a small subset of each split
    if test_mode:
        for split in dataset:
            # Ensure we don't try to take more samples than available
            max_samples = min(test_samples, len(dataset[split]))
            dataset[split] = dataset[split].select(range(max_samples))
            LOGGER.info(
                f"Test mode: reduced {split} split to {len(dataset[split])} samples"
            )

    # Initialize tokenizer
    LOGGER.info(f"Loading tokenizer from {cfg.model.configs.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.configs.model_path)

    # Set max length
    max_length = min(int(cfg.data.max_length), tokenizer.model_max_length)
    LOGGER.info(f"Using max sequence length: {max_length}")

    # Use Dataset.unique() to extract unique codes more efficiently
    LOGGER.info("Collecting unique ICD codes...")
    all_diag_codes = set()
    all_proc_codes = set()

    # Process each split separately to avoid loading entire dataset at once
    for split in dataset:
        if diagnosis_column in dataset[split].features:
            # This is more efficient than iterating through examples
            split_diag_codes = dataset[split].flatten()[diagnosis_column]
            all_diag_codes.update(
                [code for codes in split_diag_codes if codes for code in codes]
            )

        if procedure_column in dataset[split].features:
            split_proc_codes = dataset[split].flatten()[procedure_column]
            all_proc_codes.update(
                [code for codes in split_proc_codes if codes for code in codes]
            )

    all_codes = sorted(list(all_diag_codes.union(all_proc_codes)))
    LOGGER.info(f"Found {len(all_codes)} unique ICD codes")

    # Create code mappings
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}

    # NOW filter the dataset to remove examples without valid codes
    # This needs to happen AFTER code_to_idx is created
    def has_valid_codes(example):
        has_diag = (
            diagnosis_column in example
            and example[diagnosis_column]
            and any(c in code_to_idx for c in example[diagnosis_column])
        )
        has_proc = (
            procedure_column in example
            and example[procedure_column]
            and any(c in code_to_idx for c in example[procedure_column])
        )
        return has_diag or has_proc

    LOGGER.info("Filtering examples with no valid codes...")
    filtered_dataset = {}
    for split, data in dataset.items():
        filtered_data = data.filter(has_valid_codes)
        filtered_dataset[split] = filtered_data
        LOGGER.info(
            f"Split {split}: kept {len(filtered_data)}/{len(data)} examples with valid codes"
        )

    dataset = filtered_dataset

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples[TEXT_COLUMN],
            padding=False,
            truncation=True,
            max_length=max_length,
        )

        batch_size = len(examples[TEXT_COLUMN])
        labels = np.zeros((batch_size, len(all_codes)), dtype=np.float32)

        if diagnosis_column in examples:
            for i, codes in enumerate(examples[diagnosis_column]):
                if codes and isinstance(
                    codes, list
                ):  # Check that codes exists and is a list
                    # Get indices for all valid codes in one operation
                    indices = [
                        code_to_idx[code] for code in codes if code in code_to_idx
                    ]
                    if indices:
                        labels[i, indices] = 1.0

        if procedure_column in examples:
            for i, codes in enumerate(examples[procedure_column]):
                if codes and isinstance(codes, list):
                    # Get indices for all valid codes in one operation
                    indices = [
                        code_to_idx[code] for code in codes if code in code_to_idx
                    ]
                    if indices:
                        labels[i, indices] = 1.0

        tokenized["labels"] = labels.tolist()
        return tokenized

    LOGGER.info("Preprocessing dataset...")
    tokenized_dataset = {}

    for split in dataset:
        LOGGER.info(f"Processing {split} split...")
        tokenized_dataset[split] = dataset[split].map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            num_proc=8,
            remove_columns=[
                col for col in dataset[split].column_names if col != TEXT_COLUMN
            ],
            desc=f"Processing {split} split",
        )

    # Define metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Apply sigmoid and threshold
        sigmoid_preds = 1 / (1 + np.exp(-predictions))
        y_pred = (sigmoid_preds >= 0.5).astype(int)

        # Calculate metrics
        macro_f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, y_pred, average="micro", zero_division=0)
        precision = precision_score(labels, y_pred, average="micro", zero_division=0)
        recall = recall_score(labels, y_pred, average="micro", zero_division=0)

        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "precision": precision,
            "recall": recall,
        }

    LOGGER.info(f"Loading model from {cfg.model.configs.model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.configs.model_path,
        problem_type="multi_label_classification",
        num_labels=len(all_codes),
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    # Adjust epochs and other parameters for test mode
    num_epochs = 1 if test_mode else int(cfg.trainer.epochs)
    eval_steps = 10 if test_mode else 100

    # Generate a random run ID for output directory
    run_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    final_model_dir = Path("models") / run_id
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Setup training arguments with memory optimizations
    training_args = TrainingArguments(
        output_dir=str(final_model_dir),
        learning_rate=float(cfg.optimizer.configs.lr),
        per_device_train_batch_size=cfg.dataloader.max_batch_size,
        per_device_eval_batch_size=cfg.dataloader.max_batch_size,
        num_train_epochs=num_epochs,
        save_strategy="steps" if test_mode else "epoch",
        eval_strategy="steps" if test_mode else "epoch",
        eval_steps=eval_steps,
        logging_dir=str(final_model_dir / "logs"),
        logging_steps=10 if test_mode else 100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=True,
        bf16_full_eval=True,
        optim="adamw_torch",
        report_to=["wandb"],
        run_name=run_id,
    )

    # Use the efficient DataCollator that does dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    LOGGER.info("Starting training...")
    trainer.train()

    # Evaluate on test set
    LOGGER.info("Evaluating on test set...")
    trainer.evaluate(tokenized_dataset["test"])

    # Save model, tokenizer, and code mappings to final location
    LOGGER.info(f"Saving model to {final_model_dir}...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    with open(final_model_dir / "code_to_idx.json", "w") as f:
        json.dump(code_to_idx, f)
    with open(final_model_dir / "idx_to_code.json", "w") as f:
        json.dump({idx: code for code, idx in code_to_idx.items()}, f)

    LOGGER.info(f"Training complete. Model saved to {final_model_dir}")


if __name__ == "__main__":
    main()
