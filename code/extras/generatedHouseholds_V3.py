import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
import random
import json
import time
from datetime import timedelta
import numpy as np
from collections import Counter
import shutil  # For directory operations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import argparse

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic households using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Generation for area: {selected_area_code}")

# Set display options permanently
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=torch.inf)

def get_target_tensors(cross_table, hh_categories, hh_map, feature_categories, feature_map):
    y_hh = torch.zeros(num_households, dtype=torch.long, device=device)
    y_feature = torch.zeros(num_households, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross-table and categories
    household_idx = 0

    for _, row in cross_table.iterrows():
        for hh in hh_categories:
            for feature in feature_categories:
                col_name = f'{hh} {feature}'
                count = int(row.get(col_name, -1))
                if count == -1: 
                    print(col_name)
                for _ in range(count):
                    if household_idx < num_households:
                        y_hh[household_idx] = hh_map.get(hh, -1)
                        y_feature[household_idx] = feature_map.get(feature, -1)
                        household_idx += 1

    return y_hh, y_feature

# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Households.csv'))
hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Updated.csv'))
# hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity.csv'))
# hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion.csv'))
hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion_Updated.csv'))
hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_Updated.csv'))

# Define the Oxford areas
# oxford_areas = ['E02005924']
# oxford_areas = ['E02005923']
# oxford_areas = ['E02005925']

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")

ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']

# Filter the DataFrame for the specified Oxford areas
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
hhcomp_df = hhcomp_df[hhcomp_df['geography code'].isin(oxford_areas)]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'].isin(oxford_areas)]
hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'].isin(oxford_areas)]

num_households = int(hhcomp_df['total'].iloc[0])
print(f"Number of households: {num_households}")

# Preprocess household composition data
# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-1C'] + hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-1C'] + hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-1C'] + hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-1C'] + hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(['1FM', '1FC', '1FL'], axis=1)

# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-nC', '1FC-nC', '1FL-nC', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(columns=['total', 'geography code'], inplace=True)

hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
# hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-nC', '1FM-nA','1FC-0C','1FC-nC','1FC-nA','1FL-nA','1FL-nC','1H-nS','1H-nE','1H-nA', '1H-nC']

# Filter and preprocess columns
# filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]
# filtered_columns = [col for col in hhcomp_by_religion_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_religion_df = hhcomp_by_religion_df[filtered_columns]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns = ['total', 'geography code'])
hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns = ['total', 'geography code'])

# Encode the categories to indices
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
hh_map = {category: i for i, category in enumerate(hh_compositions)}

# Create household nodes with unique IDs
households_nodes = torch.arange(num_households).view(num_households, 1).to(device)

# Create nodes for ethnicity and religion categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([households_nodes, ethnicity_nodes, religion_nodes], dim=0).to(device)

# Edge index generation
def generate_edge_index(num_households):
    edge_index = []
    num_ethnicities = len(ethnicity_map)
    num_religions = len(religion_map)

    ethnicity_start_idx = num_households
    religion_start_idx = ethnicity_start_idx + num_ethnicities

    for i in range(num_households):
        # Randomly select an ethnicity and religion
        ethnicity_category = random.choice(range(ethnicity_start_idx, ethnicity_start_idx + num_ethnicities))
        religion_category = random.choice(range(religion_start_idx, religion_start_idx + num_religions))
        
        # Append edges for the selected categories
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])

    # Convert edge_index to a tensor and transpose for PyTorch Geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index
edge_index = generate_edge_index(num_households)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index).to(device)

# Enhanced GNN Model
class EnhancedGNNModelHousehold(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, mlp_hidden_dim, out_channels_hh, out_channels_ethnicity, out_channels_religion):
        super(EnhancedGNNModelHousehold, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        # Graph normalization layers
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.graph_norm3 = GraphNorm(hidden_channels)
        self.graph_norm4 = GraphNorm(hidden_channels)
        
        # Dropout layer
        # self.dropout = torch.nn.Dropout(0.1)
        
        # MLP layers for each classification target
        self.mlp_hh = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_hh)
        )
        
        self.mlp_ethnicity = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_ethnicity)
        )
        
        self.mlp_religion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_religion)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GraphSAGE layers with GraphNorm
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # x = self.conv4(x, edge_index)
        # x = self.graph_norm4(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        
        # Pass the node embeddings through the MLPs for final attribute predictions
        hh_out = self.mlp_hh(x[:num_households])
        ethnicity_out = self.mlp_ethnicity(x[:num_households])
        religion_out = self.mlp_religion(x[:num_households])
        
        return hh_out, ethnicity_out, religion_out

targets = []
targets.append(
    (
        ('hhcomp', 'religion'), 
        get_target_tensors(hhcomp_by_religion_df, hh_compositions, hh_map, religion_categories, religion_map)
    )
)
targets.append(
    (
        ('hhcomp', 'ethnicity'), 
        get_target_tensors(hhcomp_by_ethnicity_df, hh_compositions, hh_map, ethnicity_categories, ethnicity_map)
    )
)

# Hyperparameter Tuning
learning_rates = [0.001, 0.0005, 0.0001]
hidden_channel_options = [64, 128, 256]
# learning_rates = [0.001]
# hidden_channel_options = [64]
mlp_hidden_dim = 256
num_epochs = 2000
# num_epochs = 10

# Results storage
results = []
time_results = []
best_model_info = {
    'model_state': None,
    'loss': float('inf'),
    'accuracy': 0,
    'predictions': None,
    'lr': None,
    'hidden_channels': None,
    'training_time': None
}

# Function to train model
def train_model(lr, hidden_channels, num_epochs, data, targets):
    # Initialize model, optimizer, and loss functions
    model = EnhancedGNNModelHousehold(
        in_channels=node_features.size(1), 
        hidden_channels=hidden_channels, 
        mlp_hidden_dim=mlp_hidden_dim,
        out_channels_hh=len(hh_compositions), 
        out_channels_ethnicity=len(ethnicity_categories), 
        out_channels_religion=len(religion_categories)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom loss function
    def custom_loss_function(hh_out, feature_out, y_hh, y_feature):
        loss_hh = F.cross_entropy(hh_out, y_hh) 
        loss_feature = F.cross_entropy(feature_out, y_feature)
        total_loss = loss_hh + loss_feature
        return total_loss

    # Custom accuracy function for multi-task learning
    def calculate_task_accuracy(pred_1, pred_2, target_1, target_2):
        pred_1_labels = pred_1.argmax(dim=1)
        pred_2_labels = pred_2.argmax(dim=1)
        correct = ((pred_1_labels == target_1) & (pred_2_labels == target_2)).float()
        accuracy = correct.mean().item()
        return accuracy

    loss_data = {}
    accuracy_data = {}
    best_epoch_loss = float('inf')
    best_epoch_state = None
    
    # Storage for tracking metrics across epochs
    epoch_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        hh_out, ethnicity_out, religion_out = model(data)

        out = {}
        out['hhcomp'] = hh_out[:num_households]
        out['ethnicity'] = ethnicity_out[:num_households]
        out['religion'] = religion_out[:num_households]

        # Calculate loss
        loss = 0
        epoch_task_accuracies = []
        
        # Calculate losses and accuracies for all target combinations
        for i in range(len(targets)):
            current_loss = custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]],
                targets[i][1][0], targets[i][1][1]
            )
            loss += current_loss
            
            # Calculate accuracy for this task
            task_accuracy = calculate_task_accuracy(
                out[targets[i][0][0]], out[targets[i][0][1]],
                targets[i][1][0], targets[i][1][1]
            )
            epoch_task_accuracies.append(task_accuracy)

        # Calculate average accuracy for this epoch
        avg_epoch_accuracy = sum(epoch_task_accuracies) / len(epoch_task_accuracies)
        epoch_accuracies.append(avg_epoch_accuracy)
        
        # Store best epoch state
        if loss.item() < best_epoch_loss:
            best_epoch_loss = loss.item()
            best_epoch_state = model.state_dict().copy()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Store loss data for each epoch
        loss_data[epoch] = loss.item()

        # Print metrics every 100 epochs to reduce output
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {avg_epoch_accuracy:.4f}')

    # Calculate average accuracy across all epochs
    average_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

    # Load best epoch state for evaluation
    model.load_state_dict(best_epoch_state)
    
    # Evaluate predictions
    model.eval()
    with torch.no_grad():
        hh_out, ethnicity_out, religion_out = model(data)
        
        out = {}
        out['hhcomp'] = hh_out[:num_households]
        out['ethnicity'] = ethnicity_out[:num_households]
        out['religion'] = religion_out[:num_households]
        
        # Get the predicted class for each attribute by taking argmax of output logits
        hh_pred = out['hhcomp'].argmax(dim=1)
        ethnicity_pred = out['ethnicity'].argmax(dim=1)
        religion_pred = out['religion'].argmax(dim=1)
        
        # Calculate accuracy across all target combinations
        net_accuracy = 0
        final_task_accuracies = {}
        for i in range(len(targets)):
            # Get predictions for the current target combination (e.g., hhcomp+religion or hhcomp+ethnicity)
            pred_1 = out[targets[i][0][0]].argmax(dim=1)  # First attribute prediction (e.g., household composition)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)  # Second attribute prediction (e.g., religion or ethnicity)
            
            # Calculate joint accuracy - only counts as correct if BOTH predictions match the targets
            # This is a stricter metric than individual accuracy for each attribute
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1])).sum().item() / num_households
            
            # Accumulate accuracy across all target combinations
            net_accuracy += task_net_accuracy
            task_name = '_'.join(targets[i][0])
            final_task_accuracies[task_name] = task_net_accuracy * 100
        
        final_accuracy = net_accuracy / len(targets)
        
        # Print final task accuracies
        for task, acc in final_task_accuracies.items():
            print(f"{task} final accuracy: {acc:.2f}%")
        
        # Update best model info if this model performs better
        global best_model_info
        if final_accuracy > best_model_info['accuracy'] or (final_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': final_accuracy,
                'predictions': (hh_pred, ethnicity_pred, religion_pred),
                'lr': lr,
                'hidden_channels': hidden_channels
            })
    
    return best_epoch_loss, average_accuracy, final_accuracy, (hh_pred, ethnicity_pred, religion_pred)

# Run grid search over hyperparameters
total_start_time = time.time()

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, average_accuracy, final_accuracy, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            # 'average_accuracy': average_accuracy,
            'average_accuracy': final_accuracy,
            'training_time': train_time_str
        })
        
        # Store timing results
        time_results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'training_time': train_time_str
        })

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}, Average Accuracy: {average_accuracy:.4f}, Final Accuracy: {final_accuracy:.4f}")
        print(f"Training time: {train_time_str}")

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total training time: {total_training_time_str}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("\nHyperparameter tuning results:")
print(results_df)

# Print best model information
print("\nBest Model Information:")
print(f"Learning Rate: {best_model_info['lr']}")
print(f"Hidden Channels: {best_model_info['hidden_channels']}")
print(f"Best Loss: {best_model_info['loss']:.4f}")
print(f"Best Accuracy: {best_model_info['accuracy']:.4f}")

# Create output directory if it doesn't exist
# output_dir = os.path.join(current_dir, 'outputs')
output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

# Save best model information and results
# output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
# os.makedirs(output_dir, exist_ok=True)

# Save best model predictions
best_predictions = {
    'household_pred': best_model_info['predictions'][0].cpu().numpy(),
    'ethnicity_pred': best_model_info['predictions'][1].cpu().numpy(),
    'religion_pred': best_model_info['predictions'][2].cpu().numpy()
}

# Save hyperparameter results
results_df.to_csv(os.path.join(output_dir, 'generateHouseholds_results.csv'), index=False)

# Save best model configuration
best_config = {
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'loss': best_model_info['loss'],
    'accuracy': best_model_info['accuracy']
}

# Extract the best model's predictions for visualization
hh_pred, ethnicity_pred, religion_pred = best_model_info['predictions']

# Create household tensor with attributes from best model
household_tensor = torch.zeros(num_households, 4, device=device)
household_tensor[:, 0] = torch.arange(num_households, device=device)  # Household ID
household_tensor[:, 1] = hh_pred  # Household composition
household_tensor[:, 2] = ethnicity_pred  # Ethnicity
household_tensor[:, 3] = religion_pred  # Religion

# Save household tensor
household_tensor_path = os.path.join(output_dir, 'household_nodes.pt')
torch.save(household_tensor.cpu(), household_tensor_path)
print(f"\nBest model outputs saved to {output_dir}")

# Get the predicted household compositions, ethnicities, and religions
hh_comp_pred_indices = hh_pred.cpu().numpy()
ethnicity_pred_indices = ethnicity_pred.cpu().numpy()
religion_pred_indices = religion_pred.cpu().numpy()

# Convert indices to category names
hh_comp_pred_names = [hh_compositions[i] for i in hh_comp_pred_indices]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred_indices]
religion_pred_names = [religion_categories[i] for i in religion_pred_indices]

# Calculate counts of actual categories from the original data
hh_comp_actual = {}
for hh_comp in hh_compositions:
    hh_comp_actual[hh_comp] = hhcomp_df[hh_comp].iloc[0]

# Fix ethnicity and religion extraction based on column structure in the datasets
ethnicity_actual = {}
for eth in ethnicity_categories:
    ethnicity_actual[eth] = ethnicity_df[eth].iloc[0]

religion_actual = {}
for rel in religion_categories:
    religion_actual[rel] = religion_df[rel].iloc[0]

# ethnicity_actual = {}
# for eth in ethnicity_categories:
#     # Look for column that ends with the ethnicity code
#     for col in ethnicity_df.columns:
#         if col.endswith(eth) and 'count' in ethnicity_df.columns:
#             ethnicity_actual[eth] = ethnicity_df[ethnicity_df[col] == 1]['count'].sum()
#             break
#     if eth not in ethnicity_actual:
#         # Fallback if column structure is different
#         ethnicity_actual[eth] = 0

# religion_actual = {}
# for rel in religion_categories:
#     # Look for column that ends with the religion code
#     for col in religion_df.columns:
#         if col.endswith(rel) and 'count' in religion_df.columns:
#             religion_actual[rel] = religion_df[religion_df[col] == 1]['count'].sum()
#             break
#     if rel not in religion_actual:
#         # Fallback if column structure is different
#         religion_actual[rel] = 0

# print("============ Actual Values =============")
# print("\nHH Composition:")
# print(hh_comp_actual)
# print("\nEthnicity:")
# print(ethnicity_actual)
# print("\nReligion:")
# print(religion_actual)

# Calculate counts of predicted categories
hh_comp_pred = dict(Counter(hh_comp_pred_names))
ethnicity_pred = dict(Counter(ethnicity_pred_names))
religion_pred = dict(Counter(religion_pred_names))

# print("============ Predicted Values =============")
# print("\nHH Composition:")
# print(hh_comp_pred)
# print("\nEthnicity:")
# print(ethnicity_pred)
# print("\nReligion:")
# print(religion_pred)

# Normalize the actual distributions to match the total number of households in predictions
# This ensures fair comparison of relative proportions
total_actual_ethnicity = sum(ethnicity_actual.values())
total_actual_religion = sum(religion_actual.values())
total_pred = num_households

if total_actual_ethnicity > 0:
    ethnicity_actual = {k: v * total_pred / total_actual_ethnicity for k, v in ethnicity_actual.items()}
if total_actual_religion > 0:
    religion_actual = {k: v * total_pred / total_actual_religion for k, v in religion_actual.items()}

# Now create crosstable dataframes for visualization
# Reshape actual crosstables to match our format
hh_by_ethnicity_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Extract the actual counts from the crosstable dataframes
for hh in hh_compositions:
    for eth in ethnicity_categories:
        col_name = f'{hh} {eth}'
        if col_name in hhcomp_by_ethnicity_df.columns:
            hh_by_ethnicity_actual_reshaped.loc[hh, eth] = hhcomp_by_ethnicity_df[col_name].iloc[0]
    
    for rel in religion_categories:
        col_name = f'{hh} {rel}'
        if col_name in hhcomp_by_religion_df.columns:
            hh_by_religion_actual_reshaped.loc[hh, rel] = hhcomp_by_religion_df[col_name].iloc[0]

# print("============ Actual Crosstables =============")
# print("\nHH by Ethnicity:")
# print(hh_by_ethnicity_actual_reshaped)
# print("\nHH by Religion:")
# print(hh_by_religion_actual_reshaped)

# Create predicted crosstables from our predictions
hh_by_ethnicity_pred = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_pred = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Fill the predicted crosstables based on our model predictions
for i in range(len(hh_comp_pred_names)):
    hh = hh_comp_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    
    hh_by_ethnicity_pred.loc[hh, eth] += 1
    hh_by_religion_pred.loc[hh, rel] += 1

# print("============ Predicted Crosstables =============")
# print("\nHH by Ethnicity:")
# print(hh_by_ethnicity_pred)
# print("\nHH by Religion:")
# print(hh_by_religion_pred)

# Function to calculate R-squared accuracy
def calculate_r2_accuracy(generated_counts, target_counts):
    """
    Simple R² measure comparing two distributions:
    R² = 1 - (SSE / SST), with SSE = sum of squared errors
    """
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)

    sse = np.sum((gen_vals - tgt_vals) ** 2)
    sst = np.sum((tgt_vals - tgt_vals.mean()) ** 2)

    return 1.0 - sse / sst if sst > 1e-12 else 1.0

# Plotly version of individual attribute distribution plots
def plotly_attribute_distributions(attribute_dicts, categories_dict, use_log=False, filter_zero_bars=False, max_cols=3):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for multiple attributes.
    
    Parameters:
    attribute_dicts - Dictionary of attribute names to (actual, predicted) count dictionaries
    categories_dict - Dictionary of attribute names to lists of categories
    use_log - Whether to use log scale for y-axis
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    max_cols - Maximum number of columns in the subplot grid
    """
    attrs = list(attribute_dicts.keys())
    num_plots = len(attrs)
    
    # Dynamically determine columns and rows
    num_cols = min(num_plots, max_cols)
    num_rows = math.ceil(num_plots / num_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[f"{attr} Distribution" for attr in attrs],
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.15,
        vertical_spacing=0.20
    )
    
    for idx, attr_name in enumerate(attrs):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Convert to arrays
        actual_counts = np.array([actual_dict.get(cat, 0) for cat in categories])
        predicted_counts = np.array([predicted_dict.get(cat, 0) for cat in categories])
        
        # Optional log transform
        if use_log:
            actual_counts = np.log1p(actual_counts)
            predicted_counts = np.log1p(predicted_counts)
        
        # Calculate R² accuracy
        r2 = calculate_r2_accuracy(
            {cat: predicted_dict.get(cat, 0) for cat in categories},
            {cat: actual_dict.get(cat, 0) for cat in categories}
        )
        
        # Add traces
        actual_trace = go.Bar(
            x=categories,
            y=actual_counts,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=categories,
            y=predicted_counts,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update subplot title to include R²
        # fig.layout.annotations[idx].text = f"{attr_name} (R²={r2:.2f})"
        fig.layout.annotations[idx].text = f"{attr_name} - Accuracy:{r2:.2f}"
    
    # Update layout
    fig.update_layout(
        height=300 * num_rows,
        width=400 * num_cols,
        title_text="Individual Attributes: Actual vs. Predicted",
        showlegend=True,
        plot_bgcolor="white",
        barmode='group',
        margin=dict(l=40, r=40, t=80, b=50)
    )
    
    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Save the plot as HTML
    # fig.write_html(os.path.join(plots_dir, "individual_attribute_distributions.html"))
    
    # Display the plot
    fig.show()

# Plotly version of crosstable plots
def plotly_crosstable_comparison(
    actual_dfs, 
    predicted_dfs, 
    titles, 
    show_keys=False, 
    num_cols=1, 
    filter_zero_bars=True
):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for crosstables.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    show_keys - Whether to show full category key combinations (True) or numeric indices (False)
    num_cols - Number of columns in the subplot grid
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    num_rows = (num_plots + num_cols - 1) // num_cols
    
    vertical_spacing = 0.4 if show_keys else 0.2
    subplot_height = 400 if show_keys else 300
    
    if num_rows > 1:
        max_spacing = 1.0 / (num_rows - 1) - 0.01  # subtract a small margin
        vertical_spacing = min(vertical_spacing, max_spacing)
    else:
        vertical_spacing = 0
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=titles,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.15
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        actual_vals = []
        predicted_vals = []
        original_indices = []  # Store original indices from glossary
        
        sequential_index = 1  # Start from 1 to match glossary numbering
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                if not filter_zero_bars or not (a_val == 0 and p_val == 0):
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
                    original_indices.append(sequential_index)  # Keep original glossary index
                
                sequential_index += 1  # Always increment to maintain glossary alignment
        
        # Use original indices as x-axis labels
        x_labels = original_indices
        
        # Create continuous positions for bars (no gaps) but keep original indices as labels
        continuous_positions = list(range(1, len(actual_vals) + 1))
        original_indices_labels = original_indices
        
        # Calculate weighted accuracy metric
        total_actual = np.sum(actual_vals)
        accuracy = 0.0
        if total_actual > 0:
            for a_val, p_val in zip(actual_vals, predicted_vals):
                if a_val > 0:
                    accuracy += max(0, 1 - abs(p_val - a_val) / a_val) * (a_val / total_actual)
        accuracy *= 100.0
        
        # Create bar traces using continuous positions
        actual_trace = go.Bar(
            x=continuous_positions,
            y=actual_vals,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=continuous_positions,
            y=predicted_vals,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update subplot title to include accuracy
        fig.layout.annotations[idx].text = f"{titles[idx]} - Accuracy:{accuracy:.2f}%"
        
        # Update x-axis to show original indices as labels at continuous positions
        fig.update_xaxes(
            ticktext=original_indices_labels,
            tickvals=continuous_positions,
            tickangle=90,  # Angle the labels for better readability
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        height=subplot_height * num_rows + 100,  # Add extra space for titles and margins
        title_text="Crosstable Comparison: Actual vs. Predicted",
        showlegend=True,
        barmode='group',
        plot_bgcolor="white",
        margin=dict(
            b=200 if show_keys else 100,
            t=100,
            l=50,
            r=50
        )
    )
    
    fig.update_yaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Add x-axis line styling without overriding tick settings
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Save the plot as HTML
    # fig.write_html(os.path.join(plots_dir, "crosstable_comparisons.html"))
    
    # Display the plot
    fig.show()

# Plotly version of radar crosstable comparison
def plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles):
    """
    Creates radar chart subplots comparing actual vs. predicted distributions for crosstables.
    Uses numeric indices instead of category labels and shows aggregated actual vs predicted lines.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Set to one column, one plot per row
    num_cols = 1
    num_rows = num_plots
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=titles,
        specs=[[{'type': 'polar'}] for _ in range(num_rows)],
        vertical_spacing=0.1  # Increased vertical spacing between subplots
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = idx + 1
        col = 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays
        actual_vals = actual_df.values.flatten()
        predicted_vals = predicted_df.values.flatten()
        
        # Create numeric indices for the categories
        num_points = len(actual_vals)
        
        # Determine step size for labels based on number of points
        if num_points > 400:
            step_size = 16
        elif num_points > 200:
            step_size = 8
        elif num_points > 40:
            step_size = 4
        elif num_points > 30:
            step_size = 3
        elif num_points > 20:
            step_size = 2
        else:
            step_size = 1
            
        # Create labels with appropriate step size
        theta = []
        for i in range(num_points):
            if i % step_size == 0:
                theta.append(f"{i+1}")
            else:
                theta.append("")
        
        # Add the first value again to close the polygon
        actual_vals = np.append(actual_vals, actual_vals[0])
        predicted_vals = np.append(predicted_vals, predicted_vals[0])
        theta = theta + [theta[0]]
        
        # Calculate accuracy
        total_actual = np.sum(actual_vals[:-1])
        accuracy = 0.0
        if total_actual > 0:
            for a_val, p_val in zip(actual_vals[:-1], predicted_vals[:-1]):
                if a_val > 0:
                    accuracy += max(0, 1 - abs(p_val - a_val) / a_val) * (a_val / total_actual)
        accuracy *= 100.0
        
        # Create traces
        actual_trace = go.Scatterpolar(
            r=actual_vals,
            theta=theta,
            name='Actual' if idx == 0 else None,
            line=dict(color='red', width=2),
            showlegend=idx == 0
        )
        
        predicted_trace = go.Scatterpolar(
            r=predicted_vals,
            theta=theta,
            name=f'Predicted (Acc: {accuracy:.1f}%)' if idx == 0 else None,
            line=dict(color='blue', width=2),
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update subplot title to include accuracy
        fig.layout.annotations[idx].text = f"{titles[idx]} - Accuracy:{accuracy:.2f}%"
        fig.layout.annotations[idx].font.size = 14  # Reduced title font size
        fig.layout.annotations[idx].y = fig.layout.annotations[idx].y + 0.02  # Move title up slightly
    
    # Update layout with larger dimensions
    fig.update_layout(
        height=450 * num_rows,  # Slightly increased height to accommodate title spacing
        width=1000,
        title_text="Radar Chart Comparison: Actual vs. Predicted",
        title_font_size=18,  # Main title size
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=12)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(
                    actual_df.values.max(),
                    predicted_df.values.max()
                )]
            )
        ),
        margin=dict(t=120, b=80, l=100, r=100)  # Increased top margin for titles
    )
    
    # Update polar axes for each subplot
    for i in range(1, num_rows + 1):
        fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced radial axis font size
                ),
                angularaxis=dict(
                    showline=True,
                    showticklabels=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                    tickfont=dict(size=8),  # Reduced angular axis font size
                    rotation=90,
                    direction="clockwise"
                )
            ),
            row=i,
            col=1
        )
    
    # Display the plot
    fig.show()

# Create Plotly individual attribute plots
attribute_dicts = {
    'Household Composition': (hh_comp_actual, hh_comp_pred),
    'Ethnicity': (ethnicity_actual, ethnicity_pred),
    'Religion': (religion_actual, religion_pred)
}

categories_dict = {
    'Household Composition': hh_compositions,
    'Ethnicity': ethnicity_categories,
    'Religion': religion_categories
}

plotly_attribute_distributions(attribute_dicts, categories_dict, filter_zero_bars=True)

# Create Plotly crosstable plots
actual_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_actual_reshaped,
    'Household_by_Religion': hh_by_religion_actual_reshaped
}

predicted_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_pred,
    'Household_by_Religion': hh_by_religion_pred
}

titles = [
    'Household Composition x Ethnicity',
    'Household Composition x Religion'
]

plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True)

# Create Plotly radar crosstable plots
plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles)

# Create glossary tables for crosstable numeric indices
def create_crosstable_glossary(row_categories, col_categories, crosstable_name):
    """
    Creates a glossary mapping numeric indices to actual category labels for crosstables.
    
    Parameters:
    row_categories - List of row categories (e.g., household compositions)
    col_categories - List of column categories (e.g., ethnicity or religion)
    crosstable_name - Name of the crosstable for the CSV filename
    
    Returns:
    DataFrame with columns: Numeric_Index, Row_Category, Column_Category, Full_Label
    """
    glossary_data = []
    
    sequential_index = 1  # Start from 1 to match x_labels generation
    for i, row_cat in enumerate(row_categories):
        for j, col_cat in enumerate(col_categories):
            full_label = f"{row_cat} {col_cat}"
            
            glossary_data.append({
                'Sequential_Index': sequential_index,
                # 'Row_Index': i,
                # 'Column_Index': j,
                # 'Row_Category': row_cat,
                # 'Column_Category': col_cat,
                'Full_Label': full_label
            })
            
            sequential_index += 1
    
    glossary_df = pd.DataFrame(glossary_data)
    
    # Save to CSV
    glossary_path = os.path.join(output_dir, f'glossary_{crosstable_name}.csv')
    glossary_df.to_csv(glossary_path, index=False)
    print(f"Glossary for {crosstable_name} saved to: {glossary_path}")
    
    return glossary_df

# # Create glossaries for both crosstables
# print("\n" + "="*60)
# print("CREATING CROSSTABLE GLOSSARIES")
# print("="*60)

# # Glossary for Household Composition by Ethnicity
# hh_ethnicity_glossary = create_crosstable_glossary(
#     hh_compositions, 
#     ethnicity_categories, 
#     'HH_Composition_by_Ethnicity'
# )

# print(f"\nHousehold Composition by Ethnicity Glossary:")
# print(f"Total combinations: {len(hh_ethnicity_glossary)}")
# print("Sample entries:")
# print(hh_ethnicity_glossary.head(10))

# # Glossary for Household Composition by Religion
# hh_religion_glossary = create_crosstable_glossary(
#     hh_compositions, 
#     religion_categories, 
#     'HH_Composition_by_Religion'
# )

# print(f"\nHousehold Composition by Religion Glossary:")
# print(f"Total combinations: {len(hh_religion_glossary)}")
# print("Sample entries:")
# print(hh_religion_glossary.head(10))

# print(f"\nAll glossary files saved to: {output_dir}")