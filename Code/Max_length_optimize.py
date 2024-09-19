import pandas as pd
import matplotlib.pyplot as plt

# Function to load SMILES from a CSV file
def load_smiles_from_csv(file_path):
    data = pd.read_csv(file_path)
    if 'SMILES' not in data.columns:
        raise ValueError("CSV file must contain a 'SMILES' column.")
    return data['SMILES'].values.tolist()

# Function to plot SMILES length distribution
def plot_smiles_length_distribution(smiles_data):
    lengths = [len(smiles) for smiles in smiles_data]
    plt.hist(lengths, bins=range(0, max(lengths) + 1, 5), edgecolor='black')
    plt.xlabel('Length of SMILES Strings')
    plt.ylabel('Frequency')
    plt.title('Distribution of SMILES Lengths')
    plt.savefig('Smiles-dist.png')  # Save the plot
    plt.close()  # Close the plot to free up memory
    plt.show()

# Load your SMILES data
smiles_data = load_smiles_from_csv('Test_Smiles.csv')
plot_smiles_length_distribution(smiles_data)
