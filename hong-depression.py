import argparse
import pandas as pd
from pathlib import Path
import shutil
from dataclasses import dataclass
from typing import Optional
import logging
from sklearn.metrics import f1_score

from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)

from dataset import get_preprocessed_data, labels

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data classes for configuration
@dataclass
class GlobalConfig:
    dir_with_models: str = 'trained_models'
    runs: int = 1  # Changed to 1 since we're only training one model

@dataclass
class Dropout:
    att_dropout: float
    h_dropout: float
    c_dropout: float

@dataclass
class ModelInfo:
    model_type: str
    model_name: str
    model_version: Optional[str] = None
    pretrain_model: Optional[str] = None

    def get_model_path(self) -> str:
        return self.pretrain_model or self.model_name

    def description(self) -> str:
        return f'{self.model_type}\t{self.model_name}\t{self.model_version}'

    def simple_name(self) -> str:
        return self.model_name.replace('/', '_')

# Global configuration
global_config = GlobalConfig()

def get_fine_tuning_args(model_info: ModelInfo) -> tuple[ClassificationArgs, Dropout]:
    """Set up fine-tuning arguments for the model."""
    model_args = ClassificationArgs()
    model_args.learning_rate = 5e-6
    model_args.train_batch_size = 16
    model_args.num_train_epochs = 10
    model_args.evaluate_during_training_steps = 100
    model_args.max_seq_length = 300
    model_args.weight_decay = 0.1
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.output_dir = f'{global_config.dir_with_models}{model_info.model_name}_{model_info.model_version}'
    model_args.n_gpu = 0  # Set number of GPUs to 0
    model_args.fp16 = False  # Disable fp16 as it requires CUDA
    
    dropout = Dropout(0.1, 0.1, 0.1)
    return model_args, dropout

def find_best_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the best checkpoint based on F1 score."""
    if not output_dir.exists():
        return None

    progress_csv = pd.read_csv(output_dir / "training_progress_scores.csv")
    max_f1_idx = progress_csv['f1'].idxmax()
    row = progress_csv.iloc[[max_f1_idx]]
    step = row["global_step"].values[0]

    for checkpoint in output_dir.iterdir():
        if f'checkpoint-{step}' in checkpoint.name:
            return checkpoint.name

    return None

def clean(output_dir: Path, best_model: str):
    """Remove all checkpoints except the best one."""
    for checkpoint in output_dir.iterdir():
        if "checkpoint" in checkpoint.name and checkpoint.name != best_model:
            shutil.rmtree(checkpoint)

def norm(x: float) -> float:
    """Normalize a float to 3 decimal places."""
    return round(x, 3)

def macro_f1_score(y_true, y_pred):
    """Calculate macro F1 score using sklearn's f1_score function."""
    return f1_score(y_true, y_pred, average='macro')

def set_dropout(model: ClassificationModel, dropout: Dropout):
    """Set dropout rates for the model."""
    config = model.config
    config.attention_probs_dropout_prob = dropout.att_dropout
    config.hidden_dropout_prob = dropout.h_dropout
    config.classifier_dropout = dropout.c_dropout

    model.model = RobertaForSequenceClassification.from_pretrained(model.args.model_name, config=config)

def fine_tune(model_info: ModelInfo, model_args: ClassificationArgs, dropout: Dropout):
    """Fine-tune the model."""
    logger.info(f'Fine-tuning\t{model_info.description()}')

    train_data = get_preprocessed_data("train", use_shuffle=True)
    eval_data = get_preprocessed_data("dev")

    num_labels = len(labels)

    # Remove the 'with' statement
    model = ClassificationModel(
        model_info.model_type,
        model_info.get_model_path(),
        num_labels=num_labels,
        args=model_args,
        use_cuda=False,
    )
    
    set_dropout(model, dropout)
    logger.info("Starting model training...")
    model.train_model(train_data, eval_df=eval_data, f1=macro_f1_score)
    logger.info("Model training completed.")

    best_checkpoint = find_best_checkpoint(Path(model_args.output_dir))
    logger.info(f"Best checkpoint: {best_checkpoint}")
    clean(Path(model_args.output_dir), best_checkpoint)
    logger.info("Cleaned up checkpoints.")
    
    #this is the one which is optimized. let me see if this gonna be better or the other one
def get_fine_tuning_args2(model_info: ModelInfo):
    model_args = ClassificationArgs()
    
    # Learning rate: A slightly lower learning rate for more stable training
    model_args.learning_rate = 2e-5
    
    # Use a learning rate scheduler
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_metric_minimize = True
    model_args.early_stopping_patience = 3
    model_args.evaluate_during_training_steps = 200

    # Increase batch size if your GPU can handle it
    model_args.train_batch_size = 32
    model_args.eval_batch_size = 64
    
    # Adjust number of epochs
    model_args.num_train_epochs = 5
    
    # Implement gradient accumulation for larger effective batch sizes
    model_args.gradient_accumulation_steps = 2
    
    # Adjust sequence length based on your data
    model_args.max_seq_length = 512  # Assuming your data might have longer sequences
    
    # Implement weight decay for regularization
    model_args.weight_decay = 0.01
    
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.output_dir = f'{global_config.dir_with_models}{model_info.model_name}_{model_info.model_version}'
    
    # Implement warmup steps
    model_args.warmup_ratio = 0.1
    
    # Use fp16 training if your GPU supports it
    model_args.fp16 = True
    
    # Implement logging
    model_args.logging_steps = 100
    model_args.save_steps = 1000
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = True
    
    # Adjust dropout rates
    dropout = Dropout(0.1, 0.1, 0.1)
    

def main():
    logger.info("Starting the fine-tuning process...")
    
    model_info = ModelInfo(
        model_type='roberta',
        model_name='roberta-large',
        model_version='v1'
    )

    model_args, dropout = get_fine_tuning_args(model_info)
    logger.info(f"Model arguments set up: {model_args}")
    logger.info(f"Dropout rates: attention={dropout.att_dropout}, hidden={dropout.h_dropout}, classifier={dropout.c_dropout}")

    fine_tune(model_info, model_args, dropout)
    
    logger.info("Fine-tuning process completed.")

if __name__ == "__main__":
    main()