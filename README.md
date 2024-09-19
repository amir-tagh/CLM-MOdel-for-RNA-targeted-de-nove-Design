# CLM-Model-for-RNA-targeted-de-nove-Design

## Script Breakdown:
Loading and Preprocessing:

The script assumes you have a CSV file with a column SMILES containing SMILES strings.

It loads the SMILES data, tokenizes the characters in SMILES, and vectorizes them into sequences.
![Smiles-dist](https://github.com/user-attachments/assets/37a8fdbd-7ab4-4349-aed5-676ac6aff8f2)

Model Architecture:

A simple LSTM model with an embedding layer followed by a dense layer to predict the next character in a sequence.

create_model() defines the architecture.

Training:

Input (X) and output (y) are prepared by shifting the SMILES sequences.

Categorical cross-entropy is used as the loss function, and training is performed using train_model().

SMILES Generation:

Starting from a seed SMILES string, the trained model generates new SMILES characters one-by-one until the desired length is reached.

Visualization:

The generated SMILES string is converted back into a molecule using RDKit and visualized.

## Steps to Ensure GPU Execution:

Install GPU-Compatible TensorFlow: Install the TensorFlow version with GPU support:

pip install tensorflow-gpu

Check GPU Availability: TensorFlow will automatically detect and use available GPUs. You can confirm that TensorFlow is utilizing the GPU by adding these lines to the script to check for GPU availability:

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

CUDA and cuDNN: Ensure that you have the CUDA toolkit and cuDNN properly installed and configured. You can check the TensorFlow documentation for the required versions of CUDA and cuDNN compatible with your TensorFlow version:

TensorFlow GPU Setup

Run the Script: Once the environment is set up, TensorFlow should automatically utilize the GPU. When you run the script, you should see logs indicating that the GPU is being used:

Device: /device:GPU:0

## generate_smiles function:

Takes a seed SMILES string and generates new SMILES characters using the trained model.

Pads or trims the input sequence to match max_length and continues predicting characters until num_generate characters are produced.

save_generated_smiles function:

Saves the generated SMILES strings to a text file.

plot_generated_smiles function:

Plots a histogram of the lengths of the generated SMILES strings.

### How it works:
After training the model, you can provide a seed SMILES string (e.g., "CCO") to the generate_smiles function, which will generate a new SMILES string based on the trained model.

The generated SMILES is saved to a file (generated_smiles.txt) using save_generated_smiles.

The plot_generated_smiles function generates a histogram of the lengths of the generated SMILES strings to give a visual overview of the results.


