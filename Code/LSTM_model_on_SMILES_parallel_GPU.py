import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random
import matplotlib.pyplot as plt

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

