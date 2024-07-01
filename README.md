# hong-depression-detector

## Overview
This project is designed to train a machine learning model that can identify whether a post on a social network is from a depressive person or not. The model is fine-tuned using the RoBERTa architecture, leveraging depressive people's Reddit posts as the training set.

## Requirements
- Python 3.7 or later
- Required libraries (install via pip):
  - argparse
  - pandas
  - pathlib
  - shutil
  - dataclasses
  - typing
  - logging
  - sklearn
  - simpletransformers
  - transformers

## Installation
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Set up the dataset:**
   Ensure you have a dataset of depressive people's Reddit posts formatted appropriately. The dataset should be preprocessed and accessible via the `get_preprocessed_data` function.

2. **Configure the model:**
   Modify the `GlobalConfig` and `ModelInfo` data classes if necessary to match your directory structure and model preferences.

3. **Train the model:**
   Run the main script to start the fine-tuning process:
    ```bash
    python train_model.py
    ```
   The script will log the progress and save the best model checkpoint based on the F1 score.

## Script Details
### `train_model.py`
This script handles the entire process of training the model. It includes several key functions and configurations:

- **Logging setup:** Configures logging to track the progress and important information.
- **Data classes:** Defines configurations and model information using data classes.
- **Fine-tuning arguments:** Provides default arguments for fine-tuning the model, with two different configurations (`get_fine_tuning_args` and `get_fine_tuning_args2`).
- **Model training and evaluation:**
  - `fine_tune`: Handles the training process, including setting dropout rates, training the model, and cleaning up checkpoints.
  - `set_dropout`: Sets the dropout rates for different components of the model.
  - `find_best_checkpoint`: Identifies the best checkpoint based on the F1 score.
  - `clean`: Cleans up checkpoints to save disk space.
- **Helper functions:** Includes utility functions for normalization and calculating the macro F1 score.

### Fine-Tuning Arguments
Two sets of fine-tuning arguments are provided to experiment with different training configurations:
- **`get_fine_tuning_args`:** Default configuration with a learning rate of 5e-6, batch size of 16, and 10 epochs.
- **`get_fine_tuning_args2`:** Optimized configuration with a lower learning rate, early stopping, gradient accumulation, and longer sequence length.

### Running the Script
The main function initializes the model information and fine-tuning arguments, then starts the fine-tuning process.

```python
if __name__ == "__main__":
    main()
```

## Additional Information
- **Dataset:** Ensure that the `get_preprocessed_data` function is implemented correctly to load your dataset.
- **Model Saving:** The trained model and its checkpoints will be saved in the directory specified by `model_args.output_dir`.

For further customization and experimentation, modify the configurations and functions as needed. Happy training!
