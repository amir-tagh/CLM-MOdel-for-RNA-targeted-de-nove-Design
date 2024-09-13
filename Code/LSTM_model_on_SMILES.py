import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from rdkit import Chem
from rdkit.Chem import Draw
import random

# Load and preprocess the SMILES data
def load_smiles_data(file_path, num_samples=None):
    data = pd.read_csv(file_path)
    smiles_list = data['SMILES'].values[:num_samples]  # Assuming SMILES column
    return smiles_list

# Tokenize the SMILES strings
def tokenize_smiles(smiles_list):
    all_chars = sorted(set(''.join(smiles_list)))  # Get unique characters in SMILES
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    max_length = max([len(smiles) for smiles in smiles_list])
    return char_to_idx, idx_to_char, max_length

# Vectorize SMILES strings into numerical sequences
def vectorize_smiles(smiles_list, char_to_idx, max_length):
    sequences = []
    for smiles in smiles_list:
        sequence = [char_to_idx[char] for char in smiles]
        sequence = sequence + [0] * (max_length - len(sequence))  # Padding
        sequences.append(sequence)
    return np.array(sequences)

# Create LSTM model
def create_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, batch_size=64, epochs=50):
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

# Generate new SMILES strings
def generate_smiles(model, char_to_idx, idx_to_char, max_length, seed_smiles, num_generate=100):
    generated_smiles = seed_smiles
    for _ in range(num_generate):
        input_seq = [char_to_idx[char] for char in generated_smiles]
        input_seq = input_seq + [0] * (max_length - len(input_seq))  # Padding
        input_seq = np.array(input_seq).reshape(1, -1)
        prediction = model.predict(input_seq, verbose=0)
        next_idx = np.argmax(prediction)
        next_char = idx_to_char[next_idx]
        generated_smiles += next_char
        if next_char == '\n':  # End sequence
            break
    return generated_smiles

# Helper to decode generated SMILES into molecule images
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol)
    else:
        return None

if __name__ == '__main__':
    # Step 1: Load and preprocess SMILES data
    file_path = 'smiles_data.csv'  # Replace with your file path
    num_samples = 10000  # Limit the dataset size for training
    smiles_list = load_smiles_data(file_path, num_samples)

    # Step 2: Tokenization
    char_to_idx, idx_to_char, max_length = tokenize_smiles(smiles_list)

    # Step 3: Vectorize SMILES strings
    sequences = vectorize_smiles(smiles_list, char_to_idx, max_length)

    # Step 4: Prepare input-output sequences (X and y)
    X = sequences[:, :-1]  # Input (all chars except last one)
    y = sequences[:, 1:]  # Output (all chars except first one)
    y = to_categorical(y, num_classes=len(char_to_idx))  # One-hot encoding

    # Step 5: Create and train LSTM model
    model = create_model(vocab_size=len(char_to_idx), max_length=max_length - 1)
    train_model(model, X, y)

    # Step 6: Generate new SMILES strings
    seed_smiles = random.choice(smiles_list)
    print(f"Seed SMILES: {seed_smiles}")
    new_smiles = generate_smiles(model, char_to_idx, idx_to_char, max_length - 1, seed_smiles)
    print(f"Generated SMILES: {new_smiles}")

    # Step 7: Visualize generated molecule
    img = smiles_to_image(new_smiles)
    if img:
        img.show()

