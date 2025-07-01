import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
import random
import json

# Set display options permanently
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

substring_mapping = {
    'SP-Elder': '1PE',
    'SP-Adult': '1PA',
    'OF-Elder': '1FE',
    'OF-Married-0C': '1FM-0C',
    'OF-Married-2C': '1FM-2C',
    'OF-Married-ND': '1FM-nA',
    'OF-Cohabiting-0C': '1FC-0C',
    'OF-Cohabiting-2C': '1FC-2C',
    'OF-Cohabiting-ND': '1FC-nA',
    'OF-Lone-2C': '1FL-2C',
    'OF-Lone-ND': '1FL-nA',
    'OH-2C': '1H-2C',
    'OH-Student': '1H-nS',
    'OH-Elder': '1H-nE',
    'OH-Adult': '1H-nA',
}

torch.set_printoptions(edgeitems=torch.inf)

def get_target_tensors(cross_table, hh_categories, hh_map, feature_categories, feature_map):
    y_hh = torch.zeros(num_households, dtype=torch.long)
    y_feature = torch.zeros(num_households, dtype=torch.long)
    
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
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
hhcomp_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/HH_compositions_old.csv'))
hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_new.csv'))
hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/HH_composition_by_religion_new.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']

ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
religion_categories = ['C','B','H','J','M','S','OR','NR','NS']

# Filter the DataFrame for the specified Oxford areas
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
hhcomp_df = hhcomp_df[hhcomp_df['geography code'].isin(oxford_areas)]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'].isin(oxford_areas)]
hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'].isin(oxford_areas)]

# Preprocess household composition data
hhcomp_df['1FM-2C'] = hhcomp_df['1FM-1C'] + hhcomp_df['1FM-nC']
hhcomp_df['1FC-2C'] = hhcomp_df['1FC-1C'] + hhcomp_df['1FC-nC']
hhcomp_df['1FL-2C'] = hhcomp_df['1FL-1C'] + hhcomp_df['1FL-nC']
hhcomp_df['1H-2C'] = hhcomp_df['1H-1C'] + hhcomp_df['1H-nC']
hhcomp_df.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC', 'total', 'geography code'], inplace=True)
hhcomp_df = hhcomp_df.drop(['1FM', '1FC', '1FL'], axis=1)
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']

# Substring replacement for ethnicity and religion tables
for col in hhcomp_by_ethnicity_df.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            hhcomp_by_ethnicity_df.rename(columns={col: new_col}, inplace=True)
            break

for col in hhcomp_by_religion_df.columns:
    for old_substring, new_substring in substring_mapping.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            hhcomp_by_religion_df.rename(columns={col: new_col}, inplace=True)
            break

# Filter and preprocess columns
filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]
filtered_columns = [col for col in hhcomp_by_religion_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
hhcomp_by_religion_df = hhcomp_by_religion_df[filtered_columns]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns = ['total', 'geography code'])
hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns = ['total', 'geography code'])

# Mapping dictionary for religion columns
mapping_dict = {
    '1PE O': '1PE OR',
    '1PE N': '1PE NR',
    '1PA O': '1PA OR',
    '1PA N': '1PA NR',
    # ... complete this mapping ...
}
hhcomp_by_religion_df.rename(columns=mapping_dict, inplace=True)

# Encode the categories to indices
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
hh_map = {category: i for i, category in enumerate(hh_compositions)}

# Total number of households from the total column
num_households = 4852

# Create household nodes with unique IDs
households_nodes = torch.arange(num_households).view(num_households, 1)

# Create nodes for ethnicity and religion categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float)
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float)

# Combine all nodes into a single tensor
node_features = torch.cat([households_nodes, ethnicity_nodes, religion_nodes], dim=0)

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
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

# Generate edge index
edge_index = generate_edge_index(num_households)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index)

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
        self.dropout = torch.nn.Dropout(0.5)
        
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
learning_rates = [0.0005]
hidden_channel_options = [256]
mlp_hidden_dim = 256
num_epochs = 2000

# Results storage
results = []

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
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom loss function
    def custom_loss_function(hh_out, feature_out, y_hh, y_feature):
        loss_hh = F.cross_entropy(hh_out, y_hh) 
        loss_feature = F.cross_entropy(feature_out, y_feature)
        total_loss = loss_hh + loss_feature
        return total_loss

    loss_data = {}
    accuracy_data = {}

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
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Store loss data for each epoch
        loss_data[epoch] = loss.item()

    # Get final predictions
    model.eval()
    with torch.no_grad():
        hh_out, ethnicity_out, religion_out = model(data)
        hh_pred = hh_out[:num_households].argmax(dim=1)
        ethnicity_pred = ethnicity_out[:num_households].argmax(dim=1)
        religion_pred = religion_out[:num_households].argmax(dim=1)
        
        predictions = {
            'hh_pred': hh_pred,
            'ethnicity_pred': ethnicity_pred,
            'religion_pred': religion_pred
        }

    return loss.item(), model, predictions

# Run grid search over hyperparameters
best_model = None
best_predictions = None
best_loss = float('inf')

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Train the model for the current combination of hyperparameters
        final_loss, model, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
        })

        # Keep track of best model (lowest loss)
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_predictions = predictions

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("Hyperparameter tuning results:")
print(results_df)

# Save outputs in PyTorch format
outputs_dir = os.path.join(current_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# Save the results to a CSV for future reference
output_path = os.path.join(outputs_dir, 'hyperparameter_tuning_results.csv')
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Save the best trained model
if best_model is not None:
    model_path = os.path.join(outputs_dir, 'best_household_model.pt')
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model state dict saved to {model_path}")

# Save predictions
if best_predictions is not None:
    predictions_path = os.path.join(outputs_dir, 'household_predictions.pt')
    torch.save(best_predictions, predictions_path)
    print(f"Predictions saved to {predictions_path}")
    
    # Also save as structured tensor for assignHouseholds.py
    # Expected format: [household_composition, ethnicity, religion] (3 columns)
    # Where hh_composition is at index 0, ethnicity is at index 1, religion is at index 2
    household_nodes_tensor = torch.stack([
        best_predictions['hh_pred'],        # Column 0: household composition
        best_predictions['ethnicity_pred'], # Column 1: ethnicity
        best_predictions['religion_pred']   # Column 2: religion
    ], dim=1)
    
    household_nodes_path = os.path.join(outputs_dir, 'household_nodes.pt')
    torch.save(household_nodes_tensor, household_nodes_path)
    print(f"Household nodes tensor saved to {household_nodes_path}")

# Save data and other relevant tensors
data_artifacts = {
    'data': data,
    'targets': targets,
    'node_features': node_features,
    'edge_index': edge_index,
    'ethnicity_categories': ethnicity_categories,
    'religion_categories': religion_categories,
    'hh_compositions': hh_compositions,
    'ethnicity_map': ethnicity_map,
    'religion_map': religion_map,
    'hh_map': hh_map,
    'num_households': num_households
}
artifacts_path = os.path.join(outputs_dir, 'household_data_artifacts.pt')
torch.save(data_artifacts, artifacts_path)
print(f"Data artifacts saved to {artifacts_path}")