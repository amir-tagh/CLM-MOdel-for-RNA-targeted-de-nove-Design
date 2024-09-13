import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

# Function to generate batches of data for training
def data_generator(data, batch_size, max_length, char_to_int):
    while True:
        X = np.zeros((batch_size, max_length), dtype=np.int32)
        y = np.zeros((batch_size, len(char_to_int)), dtype=np.float32)
        for i in range(batch_size):
            random_idx = random.randint(0, len(data) - max_length - 1)
            X[i, :] = [char_to_int[char] for char in data[random_idx:random_idx + max_length]]
            y[i, char_to_int[data[random_idx + max_length]]] = 1
        yield X, y

# Function to create the model
def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

# Function to generate SMILES from a seed string
def generate_smiles(model, seed, char_to_int, int_to_char, max_length, num_generate=100):
    generated = seed
    seed_encoded = [char_to_int[char] for char in seed]
    
    for _ in range(num_generate):
        # Pad or trim seed to match max_length
        input_sequence = np.zeros((1, max_length), dtype=np.int32)
        input_sequence[0, -len(seed_encoded):] = seed_encoded[:max_length]

        # Predict next character probabilities
        prediction = model.predict(input_sequence, verbose=0)
        next_char_idx = np.argmax(prediction[0])
        
        # Add predicted character to the sequence
        next_char = int_to_char[next_char_idx]
        generated += next_char

        # Update the seed with the predicted character
        seed_encoded.append(next_char_idx)
        seed_encoded = seed_encoded[1:]  # Keep the last max_length chars

    return generated

# Function to save generated SMILES to a file
def save_generated_smiles(smiles_list, filename):
    with open(filename, 'w') as file:
        for smiles in smiles_list:
            file.write(f"{smiles}\n")

# Function to plot the generated SMILES strings
def plot_generated_smiles(smiles_list):
    lengths = [len(smiles) for smiles in smiles_list]
    plt.hist(lengths, bins=20, color='blue', alpha=0.7)
    plt.title('Distribution of Generated SMILES Lengths')
    plt.xlabel('SMILES Length')
    plt.ylabel('Frequency')
    plt.show()

# Help function to explain usage of the script
def print_help():
    help_message = """
    This script trains a character-level LSTM model on a library of SMILES strings and generates new molecules.
    
    Usage:
        1. Train the model:
           - The model is trained using SMILES data, and the weights are saved as 'best_model.h5'.
           - The data should be a string of concatenated SMILES characters.
        
        2. Generate new SMILES:
           - After training, you can generate new SMILES strings using a seed SMILES.
           - Use the `generate_smiles` function to generate new SMILES.
        
        3. Save the generated SMILES:
           - The generated SMILES strings can be saved to a text file using `save_generated_smiles`.
        
        4. Plot the SMILES distribution:
           - The lengths of the generated SMILES can be plotted using `plot_generated_smiles`.

    Arguments:
        - vocab_size: Vocabulary size (unique characters in the SMILES dataset)
        - max_length: Maximum sequence length for training and generation
        - batch_size: Batch size for training
        - epochs: Number of training epochs
        - seed_smiles: The seed SMILES string for generating new SMILES
        - num_generate: Number of characters to generate

    Example Usage:
        - python script.py --seed_smiles "CCO" --num_generate 50

    """
    print(help_message)

# Prepare multi-GPU strategy
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs available: {len(gpus)}")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Running on {strategy.num_replicas_in_sync} GPUs")
else:
    strategy = tf.distribute.get_strategy()  # Default strategy for a single GPU or CPU

# Training parameters
vocab_size = 50  # Example vocabulary size (needs to match the actual dataset)
max_length = 100  # Example sequence length
batch_size = 256  # Batch size for training
epochs = 100  # Number of training epochs

# Sample SMILES string data (replace with actual SMILES data)
data = 'CCOCCOCNCCOCCN'  # This is just placeholder data; replace with the actual SMILES dataset
chars = sorted(list(set(data)))  # Unique characters in the dataset
char_to_int = {c: i for i, c in enumerate(chars)}  # Character to integer mapping
int_to_char = {i: c for c, i in char_to_int.items()}  # Integer to character mapping

# Data generator for training
train_data_generator = data_generator(data, batch_size, max_length, char_to_int)

# Multi-GPU strategy scope
with strategy.scope():
    model = create_model(len(char_to_int), max_length)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Model summary
model.summary()

# ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='loss')

# Train the model using the data generator
model.fit(train_data_generator, steps_per_epoch=1000, epochs=epochs, callbacks=[checkpoint])

# Example usage: Generate SMILES from a seed
seed_smiles = 'CCO'  # Example seed SMILES
num_generate = 50  # Number of characters to generate
generated_smiles = generate_smiles(model, seed_smiles, char_to_int, int_to_char, max_length, num_generate)

# Save the generated SMILES to a file
save_generated_smiles([generated_smiles], 'generated_smiles.txt')

# Plot the generated SMILES length distribution
plot_generated_smiles([generated_smiles])

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Train an LSTM model and generate SMILES")
parser.add_argument("--help", action="store_true", help="Show help message")
parser.add_argument("--seed_smiles", type=str, help="Seed SMILES string for generation")
parser.add_argument("--num_generate", type=int, default=50, help="Number of characters to generate")

args = parser.parse_args()

# Show help message if requested
if args.help:
    print_help()

