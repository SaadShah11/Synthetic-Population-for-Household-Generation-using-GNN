import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm, GATv2Conv, LayerNorm, JumpingKnowledge
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
oxford_areas = ['E02005924']
# oxford_areas = ['E02005923']
# oxford_areas = ['E02005925']

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
        
        from torch_geometric.nn import SAGEConv, GraphNorm
        
        # Using GraphSAGE as the primary model - proven to work well in the original
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        # Graph normalization - using what worked in the original
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.graph_norm3 = GraphNorm(hidden_channels)
        self.graph_norm4 = GraphNorm(hidden_channels)
        
        # Task-specific attention - keeping this improvement
        self.task_attention = torch.nn.Parameter(torch.ones(3, hidden_channels))
        
        # MLP layers with LayerNorm but no dropout
        self.mlp_hh = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_hh)
        )
        
        self.mlp_ethnicity = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_ethnicity)
        )
        
        self.mlp_religion = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_religion)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1: GraphSAGE with GraphNorm - following the original successful pattern
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x)
        x = F.relu(x)
        
        # Layer 2: GraphSAGE with GraphNorm
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x)
        x = F.relu(x)
        
        # Layer 3: GraphSAGE with GraphNorm
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        x = F.relu(x)
        
        # Layer 4: GraphSAGE with GraphNorm - adding the 4th layer back
        x = self.conv4(x, edge_index)
        x = self.graph_norm4(x)
        x = F.relu(x)
        
        # Extract household nodes
        household_nodes = x[:num_households]
        
        # Apply task-specific attention - keeping this improvement
        attn = F.softmax(self.task_attention, dim=1)
        
        # Task 1: Household composition prediction with attention
        hh_features = household_nodes * attn[0]
        hh_out = self.mlp_hh(hh_features)
        
        # Task 2: Ethnicity prediction with attention
        ethnicity_features = household_nodes * attn[1]
        ethnicity_out = self.mlp_ethnicity(ethnicity_features)
        
        # Task 3: Religion prediction with attention
        religion_features = household_nodes * attn[2]
        religion_out = self.mlp_religion(religion_features)
        
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
# learning_rates = [0.001, 0.0005, 0.0001]
# hidden_channel_options = [64, 128, 256]
learning_rates = [0.001]
hidden_channel_options = [64]
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

    loss_data = {}
    accuracy_data = {}
    best_epoch_loss = float('inf')
    best_epoch_state = None

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
        for i in range(len(targets)):
            loss += custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]],
                targets[i][1][0], targets[i][1][1]
            )
        
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
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

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
        for i in range(len(targets)):
            # Get predictions for the current target combination (e.g., hhcomp+religion or hhcomp+ethnicity)
            pred_1 = out[targets[i][0][0]].argmax(dim=1)  # First attribute prediction (e.g., household composition)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)  # Second attribute prediction (e.g., religion or ethnicity)
            
            # Calculate joint accuracy - only counts as correct if BOTH predictions match the targets
            # This is a stricter metric than individual accuracy for each attribute
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1])).sum().item() / num_households
            
            # Accumulate accuracy across all target combinations
            net_accuracy += task_net_accuracy
        
        avg_accuracy = net_accuracy / len(targets)
        
        # Update best model info if this model performs better
        global best_model_info
        if avg_accuracy > best_model_info['accuracy'] or (avg_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': avg_accuracy,
                'predictions': (hh_pred, ethnicity_pred, religion_pred),
                'lr': lr,
                'hidden_channels': hidden_channels
            })
    
    return best_epoch_loss, avg_accuracy, (hh_pred, ethnicity_pred, religion_pred)

# Run grid search over hyperparameters
total_start_time = time.time()

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, avg_accuracy, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            'average_accuracy': avg_accuracy,
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
        print(f"Final Loss: {final_loss}, Average Accuracy: {avg_accuracy}")
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

# Save best model information and results
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

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
    # Look for column that ends with the ethnicity code
    for col in ethnicity_df.columns:
        if col.endswith(eth) and 'count' in ethnicity_df.columns:
            ethnicity_actual[eth] = ethnicity_df[ethnicity_df[col] == 1]['count'].sum()
            break
    if eth not in ethnicity_actual:
        # Fallback if column structure is different
        ethnicity_actual[eth] = 0

religion_actual = {}
for rel in religion_categories:
    # Look for column that ends with the religion code
    for col in religion_df.columns:
        if col.endswith(rel) and 'count' in religion_df.columns:
            religion_actual[rel] = religion_df[religion_df[col] == 1]['count'].sum()
            break
    if rel not in religion_actual:
        # Fallback if column structure is different
        religion_actual[rel] = 0

# If counts are still zero, try simpler method by summing total distribution to match household count
if sum(ethnicity_actual.values()) == 0:
    # Extract count distribution from cross table by summing columns for each ethnicity
    for eth in ethnicity_categories:
        total_eth = 0
        for col in hhcomp_by_ethnicity_df.columns:
            if f' {eth}' in col:
                total_eth += hhcomp_by_ethnicity_df[col].iloc[0]
        ethnicity_actual[eth] = total_eth

if sum(religion_actual.values()) == 0:
    # Extract count distribution from cross table by summing columns for each religion
    for rel in religion_categories:
        total_rel = 0
        for col in hhcomp_by_religion_df.columns:
            if f' {rel}' in col:
                total_rel += hhcomp_by_religion_df[col].iloc[0]
        religion_actual[rel] = total_rel

# Calculate counts of predicted categories
hh_comp_pred = dict(Counter(hh_comp_pred_names))
ethnicity_pred = dict(Counter(ethnicity_pred_names))
religion_pred = dict(Counter(religion_pred_names))

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
    
    vertical_spacing = 0.6 if show_keys else 0.4
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
        
        # Flatten the dataframes to create 1D arrays for bar charts
        # We'll create key strings from the row and column indices
        actual_vals = []
        predicted_vals = []
        all_keys = []
        
        for i, row_idx in enumerate(actual_df.index):
            for j, col_idx in enumerate(actual_df.columns):
                key = f"{row_idx} {col_idx}" if show_keys else f"{i},{j}"
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                if not filter_zero_bars or not (a_val == 0 and p_val == 0):
                    all_keys.append(key)
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
        
        # Create sequential x-axis labels
        x_labels = list(range(1, len(actual_vals) + 1))
        
        # Calculate weighted accuracy metric
        total_actual = np.sum(actual_vals)
        accuracy = 0.0
        if total_actual > 0:
            for a_val, p_val in zip(actual_vals, predicted_vals):
                if a_val > 0:
                    accuracy += max(0, 1 - abs(p_val - a_val) / a_val) * (a_val / total_actual)
        accuracy *= 100.0
        
        # Create bar traces
        actual_trace = go.Bar(
            x=x_labels,
            y=actual_vals,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        predicted_trace = go.Bar(
            x=x_labels,
            y=predicted_vals,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0  # Only show legend for first subplot
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update subplot title to include accuracy
        # fig.layout.annotations[idx].text = f"{titles[idx]} - Accuracy: {accuracy:.2f}%"
        fig.layout.annotations[idx].text = f"{titles[idx]} - Accuracy:{accuracy:.2f}%"
    
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
    
    fig.update_xaxes(
        tickcolor='black',
        ticks="outside",
        tickwidth=2,
        showline=True,
        linecolor='black',
        linewidth=2,
        tickmode='linear',
        tick0=1,
        dtick=1
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
    # fig.write_html(os.path.join(plots_dir, "crosstable_comparisons.html"))
    
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
    'Household Composition by Ethnicity',
    'Household Composition by Religion'
]

plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True)