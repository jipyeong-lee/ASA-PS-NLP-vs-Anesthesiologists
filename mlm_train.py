import warnings
warnings.filterwarnings('ignore')  # Ignore warning messages

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,  # Class for loading model configurations
    AutoModelForMaskedLM,  # Class for loading pre-trained models for masked language modeling
    AutoTokenizer,  # Class for loading tokenizers
    DataCollatorForLanguageModeling,  # Class for preparing data collator for language modeling
    TrainingArguments,  # Class for defining training arguments
    Trainer  # Class for managing the training process
)

from datasets import load_from_disk  # Function for loading datasets from disk
from torchsampler import ImbalancedDatasetSampler  # Sampler for handling imbalanced datasets

model_name = "yikuan8/Clinical-BigBird"  # Name of the pre-trained model to use

# Specify the dataset path
dataset_path = './data/resident_tuning_4096'
train_dataset = load_from_disk(dataset_path=dataset_path)  # Load dataset from the specified path

# Initialize model and tokenizer
config = AutoConfig.from_pretrained(model_name)  # Load model configuration
model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)  # Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # Load tokenizer

# Prepare data collator for language modeling with a specified MLM probability
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./ClinicalBigBird_mlm',  # Directory for saving model checkpoints
    overwrite_output_dir=True,  # Overwrite the content of the output directory
    do_train=True,  # Enable training
    per_device_train_batch_size=2,  # Batch size per device during training
    learning_rate=5e-7,  # Learning rate
    num_train_epochs=20,  # Number of training epochs
    seed=42,  # Random seed for reproducibility
    data_seed=42,  # Data seed for reproducibility
    bf16=True,  # Use bfloat16 precision
    dataloader_num_workers=16,  # Number of subprocesses to use for data loading
    save_total_limit=2,  # Limit the total amount of checkpoints
    ddp_find_unused_parameters=True,  # Find unused parameters for DDP
    gradient_accumulation_steps=32,  # Number of updates steps to accumulate before performing a backward/update pass
    lr_scheduler_type='cosine',  # Learning rate scheduler type
    torch_compile=True,  # Enable Torch compile
    torch_compile_backend='inductor',  # Set Torch compile backend
    torch_compile_mode='default',  # Set Torch compile mode
    report_to='none'  # Disable reporting to any service
)

# Define a custom trainer to handle imbalanced dataset sampling
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        def get_label(dataset):
            return dataset["label"]

        # Set up an imbalanced dataset sampler
        train_sampler = ImbalancedDatasetSampler(
            train_dataset, callback_get_label=get_label
        )

        # DataLoader parameters
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": train_sampler
        }

        return DataLoader(train_dataset, **dataloader_params)

# Instantiate the custom trainer and start training
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()  # Start the training process

# Uncomment the following line to save the model after training
# model.save_pretrained('./ClinicalBigBird_mlm/best_model')