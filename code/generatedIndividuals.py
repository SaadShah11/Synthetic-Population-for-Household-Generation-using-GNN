import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
from torch.nn import CrossEntropyLoss
import random
import time
from datetime import timedelta
import json
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

def get_target_tensors(cross_table, feature_1_categories, feature_1_map, feature_2_categories, feature_2_map, feature_3_categories, feature_3_map):
    y_feature_1 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_2 = torch.zeros(num_persons, dtype=torch.long, device=device)
    y_feature_3 = torch.zeros(num_persons, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross table and feature categories
    person_idx = 0
    for _, row in cross_table.iterrows():
        for feature_1 in feature_1_categories:
            for feature_2 in feature_2_categories:
                for feature_3 in feature_3_categories:
                    col_name = f'{feature_1} {feature_2} {feature_3}'
                    count = int(row.get(col_name, 0))
                    for _ in range(count):
                        if person_idx < num_persons:
                            y_feature_1[person_idx] = feature_1_map.get(feature_1, -1)
                            y_feature_2[person_idx] = feature_2_map.get(feature_2, -1)
                            y_feature_3[person_idx] = feature_3_map.get(feature_3, -1)
                            person_idx += 1

    return (y_feature_1, y_feature_2, y_feature_3)


# Load the data from individual tables
current_dir = os.path.dirname(os.path.abspath(__file__))
age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Age_Perfect_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Marital.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/EthnicityBySexByAge.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ReligionbySexbyAge.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/MaritalbySexbyAgeModified.csv'))
# age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
# sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
# ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
# religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
# marital_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Marital.csv'))
# ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))
# religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/religion_by_sex_by_age.csv'))
# marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/marital_by_sex_by_age.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']
# oxford_areas = ['E02005923']
# oxford_areas = ['E02005925']

# Filter the DataFrame for the specified Oxford areas
age_df = age_df[age_df['geography code'].isin(oxford_areas)]
sex_df = sex_df[sex_df['geography code'].isin(oxford_areas)]
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
marital_df = marital_df[marital_df['geography code'].isin(oxford_areas)]
ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'].isin(oxford_areas)]
religion_by_sex_by_age_df = religion_by_sex_by_age_df[religion_by_sex_by_age_df['geography code'].isin(oxford_areas)]
marital_by_sex_by_age_df = marital_by_sex_by_age_df[marital_by_sex_by_age_df['geography code'].isin(oxford_areas)]

# Define the age groups, sex categories, and ethnicity categories
age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
# age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_44', '45_59', '60_64', '65_74', '75_84', '85_89', '90+']
sex_categories = ['M', 'F']
# ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
# religion_categories = ['C','B','H','J','M','S','OR','NR','NS']
ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']

# Encode the categories to indices
age_map = {category: i for i, category in enumerate(age_groups)}
sex_map = {category: i for i, category in enumerate(sex_categories)}
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
marital_map = {category: i for i, category in enumerate(marital_categories)}

# Total number of persons from the total column
num_persons = int(age_df['total'].sum())

print(f"Total number of persons: {num_persons}")

# Create person nodes with unique IDs
person_nodes = torch.arange(num_persons).view(num_persons, 1).to(device)

# Create nodes for age categories
age_nodes = torch.tensor([[age_map[age]] for age in age_groups], dtype=torch.float).to(device)

# Create nodes for sex categories
sex_nodes = torch.tensor([[sex_map[sex]] for sex in sex_categories], dtype=torch.float).to(device)

# Create nodes for ethnicity categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)

# Create nodes for religion categories
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Create nodes for marital categories
marital_nodes = torch.tensor([[marital_map[marital]] for marital in marital_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([person_nodes, age_nodes, sex_nodes, ethnicity_nodes, religion_nodes, marital_nodes], dim=0).to(device)

# Calculate the distribution for age categories
age_probabilities = age_df.drop(columns = ["geography code", "total"]) / num_persons
sex_probabilities = sex_df.drop(columns = ["geography code", "total"]) / num_persons
ethnicity_probabilities = ethnicity_df.drop(columns = ["geography code", "total"]) / num_persons
religion_probabilities = religion_df.drop(columns = ["geography code", "total"]) / num_persons
marital_probabilities = marital_df.drop(columns = ["geography code", "total"]) / num_persons

# New function to generate edge index
def generate_edge_index(num_persons):
    edge_index = []
    age_start_idx = num_persons
    sex_start_idx = age_start_idx + len(age_groups)
    ethnicity_start_idx = sex_start_idx + len(sex_categories)
    religion_start_idx = ethnicity_start_idx + len(ethnicity_categories)
    marital_start_idx = religion_start_idx + len(religion_categories)

    # Convert the probability series to a list of probabilities for sampling
    age_prob_list = age_probabilities.values.tolist()[0]
    sex_prob_list = sex_probabilities.values.tolist()[0]
    ethnicity_prob_list = ethnicity_probabilities.values.tolist()[0]
    religion_prob_list = religion_probabilities.values.tolist()[0]
    marital_prob_list = marital_probabilities.values.tolist()[0]

    for i in range(num_persons):
        # Sample the categories using weighted random sampling
        age_category = random.choices(range(age_start_idx, sex_start_idx), weights=age_prob_list, k=1)[0]
        sex_category = random.choices(range(sex_start_idx, ethnicity_start_idx), weights=sex_prob_list, k=1)[0]
        ethnicity_category = random.choices(range(ethnicity_start_idx, religion_start_idx), weights=ethnicity_prob_list, k=1)[0]
        religion_category = random.choices(range(religion_start_idx, marital_start_idx), weights=religion_prob_list, k=1)[0]
        marital_category = random.choices(range(marital_start_idx, marital_start_idx + len(marital_categories)), weights=marital_prob_list, k=1)[0]
        
        # Append edges for each category
        edge_index.append([i, age_category])
        edge_index.append([i, sex_category])
        edge_index.append([i, ethnicity_category])
        edge_index.append([i, religion_category])
        edge_index.append([i, marital_category])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index using the new function
edge_index = generate_edge_index(num_persons)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index).to(device)

# Get target tensors
targets = []
targets.append(
    (
        ('sex', 'age', 'ethnicity'), 
        get_target_tensors(ethnic_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, ethnicity_categories, ethnicity_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'marital'), 
        get_target_tensors(marital_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, marital_categories, marital_map)
    )
)
targets.append(
    (
        ('sex', 'age', 'religion'), 
        get_target_tensors(religion_by_sex_by_age_df, sex_categories, sex_map, age_groups, age_map, religion_categories, religion_map)
    )
)

class EnhancedGNNModelWithMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, mlp_hidden_dim, out_channels_age, out_channels_sex, out_channels_ethnicity, out_channels_religion, out_channels_marital, dropout_rate=0.5):
        super(EnhancedGNNModelWithMLP, self).__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        
        # Graph normalization
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.graph_norm3 = GraphNorm(hidden_channels)
        self.graph_norm4 = GraphNorm(hidden_channels)
        
        # Task-specific attention parameters
        self.task_attention_age = torch.nn.Parameter(torch.ones(hidden_channels))
        self.task_attention_sex = torch.nn.Parameter(torch.ones(hidden_channels))
        self.task_attention_ethnicity = torch.nn.Parameter(torch.ones(hidden_channels))
        self.task_attention_religion = torch.nn.Parameter(torch.ones(hidden_channels))
        self.task_attention_marital = torch.nn.Parameter(torch.ones(hidden_channels))
        
        # MLP layers with LayerNorm for each output attribute
        self.mlp_age = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_age)
        )
        
        self.mlp_sex = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_sex)
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
        
        self.mlp_marital = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.LayerNorm(mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_marital)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GraphSAGE layers with GraphNorm
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        x = F.relu(x)
        
        x = self.conv4(x, edge_index)
        x = self.graph_norm4(x)
        x = F.relu(x)
        
        # Extract person nodes
        person_features = x[:num_persons]
        
        # Apply task-specific attention for each attribute
        age_features = person_features * F.softmax(self.task_attention_age, dim=0)
        sex_features = person_features * F.softmax(self.task_attention_sex, dim=0)
        ethnicity_features = person_features * F.softmax(self.task_attention_ethnicity, dim=0)
        religion_features = person_features * F.softmax(self.task_attention_religion, dim=0)
        marital_features = person_features * F.softmax(self.task_attention_marital, dim=0)
        
        # Pass through task-specific MLPs
        age_out = self.mlp_age(age_features)
        sex_out = self.mlp_sex(sex_features)
        ethnicity_out = self.mlp_ethnicity(ethnicity_features)
        religion_out = self.mlp_religion(religion_features)
        marital_out = self.mlp_marital(marital_features)
        
        return age_out, sex_out, ethnicity_out, religion_out, marital_out

# Custom loss function
def custom_loss_function(first_out, second_out, third_out, y_first, y_second, y_third):
    first_pred = first_out.argmax(dim=1)
    second_pred = second_out.argmax(dim=1)
    third_pred = third_out.argmax(dim=1)
    loss_first = F.cross_entropy(first_out, y_first)
    loss_second = F.cross_entropy(second_out, y_second)
    loss_third = F.cross_entropy(third_out, y_third)
    total_loss = loss_first + loss_second + loss_third
    return total_loss

# Define the hyperparameters to tune
learning_rates = [0.001]
hidden_channel_options = [64]
# learning_rates = [0.001, 0.0005, 0.0001]
# hidden_channel_options = [64, 128, 256]
mlp_hidden_dim = 128
num_epochs = 2500
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

# Define a function to train the model
def train_model(lr, hidden_channels, num_epochs, data, targets):
    # Initialize model, optimizer, and loss functions
    model = EnhancedGNNModelWithMLP(
        in_channels=node_features.size(1),
        hidden_channels=hidden_channels,
        mlp_hidden_dim=mlp_hidden_dim,
        out_channels_age=len(age_groups),
        out_channels_sex=len(sex_categories),
        out_channels_ethnicity=len(ethnicity_categories),
        out_channels_religion=len(religion_categories),
        out_channels_marital=len(marital_categories)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Track best epoch state
    best_epoch_loss = float('inf')
    best_epoch_state = None
    loss_data = {}
    accuracy_data = {}

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)

        out = {}
        out['age'] = age_out[:num_persons]  # Only take person nodes' outputs
        out['sex'] = sex_out[:num_persons]
        out['ethnicity'] = ethnicity_out[:num_persons]
        out['religion'] = religion_out[:num_persons]
        out['marital'] = marital_out[:num_persons]

        loss = 0
        for i in range(len(targets)):
            loss += custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
                targets[i][1][0], targets[i][1][1], targets[i][1][2]
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

    # Evaluate accuracy after training
    model.eval()
    with torch.no_grad():
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
        
        out = {}
        out['age'] = age_out[:num_persons]
        out['sex'] = sex_out[:num_persons]
        out['ethnicity'] = ethnicity_out[:num_persons]
        out['religion'] = religion_out[:num_persons]
        out['marital'] = marital_out[:num_persons]
        
        age_pred = out['age'].argmax(dim=1)
        sex_pred = out['sex'].argmax(dim=1)
        ethnicity_pred = out['ethnicity'].argmax(dim=1)
        religion_pred = out['religion'].argmax(dim=1)
        marital_pred = out['marital'].argmax(dim=1)

        # Calculate net accuracy across all tasks
        net_accuracy = 0
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            
            # Calculate joint accuracy - only counts as correct if ALL THREE predictions match the targets
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & 
                                (pred_2 == targets[i][1][1]) & 
                                (pred_3 == targets[i][1][2])).sum().item() / num_persons
            
            net_accuracy += task_net_accuracy
        
        avg_accuracy = net_accuracy / len(targets)
        
        # Update best model info if this model performs better
        global best_model_info
        if avg_accuracy > best_model_info['accuracy'] or (avg_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': avg_accuracy,
                'predictions': (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred),
                'lr': lr,
                'hidden_channels': hidden_channels
            })

        # Return the final loss and averaged net accuracy
        return best_epoch_loss, avg_accuracy, (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred)

# Run the grid search over hyperparameters
total_start_time = time.time()
time_results = []

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

# Create output directory if it doesn't exist
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Save best model state
# torch.save(best_model_info['model_state'], os.path.join(output_dir, 'best_individual_model_state.pt'))

# Save best model predictions
best_predictions = {
    'sex_pred': best_model_info['predictions'][0].cpu().numpy(),
    'age_pred': best_model_info['predictions'][1].cpu().numpy(),
    'ethnicity_pred': best_model_info['predictions'][2].cpu().numpy(),
    'religion_pred': best_model_info['predictions'][3].cpu().numpy(),
    'marital_pred': best_model_info['predictions'][4].cpu().numpy()
}
# np.save(os.path.join(output_dir, 'best_individual_model_predictions.npy'), best_predictions)

# Save hyperparameter results
results_df.to_csv(os.path.join(output_dir, 'generateIndividuals_results.csv'), index=False)

# Save best model configuration
best_config = {
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'loss': best_model_info['loss'],
    'accuracy': best_model_info['accuracy']
}
# with open(os.path.join(output_dir, 'best_individual_model_config.json'), 'w') as f:
#     json.dump(best_config, f, indent=4)

# Extract the best model's predictions for visualization
sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred = best_model_info['predictions']

# Create person tensor with attributes
# Format: [person_id, age, religion, ethnicity, marital, sex]
person_tensor = torch.zeros(num_persons, 6, device=device)
person_tensor[:, 0] = torch.arange(num_persons, device=device)  # Person ID
person_tensor[:, 1] = age_pred  # Age
person_tensor[:, 2] = religion_pred  # Religion
person_tensor[:, 3] = ethnicity_pred  # Ethnicity
person_tensor[:, 4] = marital_pred  # Marital status
person_tensor[:, 5] = sex_pred  # Sex

# Save person tensor
person_tensor_path = os.path.join(output_dir, 'person_nodes.pt')
torch.save(person_tensor.cpu(), person_tensor_path)
print(f"\nBest model outputs saved to {output_dir}")


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
    # fig.write_html(os.path.join(output_dir, "individual_attribute_distributions.html"))
    
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
    # fig.write_html(os.path.join(output_dir, "crosstable_comparisons.html"))
    
    # Display the plot
    fig.show()

# Prepare data for visualization
# Convert model predictions to category names
sex_pred_names = [sex_categories[i] for i in sex_pred.cpu().numpy()]
age_pred_names = [age_groups[i] for i in age_pred.cpu().numpy()]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred.cpu().numpy()]
religion_pred_names = [religion_categories[i] for i in religion_pred.cpu().numpy()]
marital_pred_names = [marital_categories[i] for i in marital_pred.cpu().numpy()]

# Calculate actual distributions
sex_actual = {}
age_actual = {}
ethnicity_actual = {}
religion_actual = {}
marital_actual = {}

# Extract counts from the original data frames
for sex in sex_categories:
    col_name = f'count-{sex}'
    if col_name in sex_df.columns:
        sex_actual[sex] = sex_df[col_name].iloc[0]
    else:
        # Fallback to searching through columns
        for col in sex_df.columns:
            if col.endswith(sex) and col != 'geography code' and col != 'total':
                sex_actual[sex] = sex_df[col].iloc[0]
                break
        if sex not in sex_actual:
            sex_actual[sex] = 0

for age in age_groups:
    col_name = f'count-{age}'
    if col_name in age_df.columns:
        age_actual[age] = age_df[col_name].iloc[0]
    else:
        # Fallback to searching through columns
        for col in age_df.columns:
            if col.endswith(age) and col != 'geography code' and col != 'total':
                age_actual[age] = age_df[col].iloc[0]
                break
        if age not in age_actual:
            age_actual[age] = 0

for eth in ethnicity_categories:
    col_name = f'count-{eth}'
    if col_name in ethnicity_df.columns:
        ethnicity_actual[eth] = ethnicity_df[col_name].iloc[0]
    else:
        # Fallback to searching through columns
        for col in ethnicity_df.columns:
            if col.endswith(eth) and col != 'geography code' and col != 'total':
                ethnicity_actual[eth] = ethnicity_df[col].iloc[0]
                break
        if eth not in ethnicity_actual:
            ethnicity_actual[eth] = 0

for rel in religion_categories:
    col_name = f'count-{rel}'
    if col_name in religion_df.columns:
        religion_actual[rel] = religion_df[col_name].iloc[0]
    else:
        # Fallback to searching through columns
        for col in religion_df.columns:
            if col.endswith(rel) and col != 'geography code' and col != 'total':
                religion_actual[rel] = religion_df[col].iloc[0]
                break
        if rel not in religion_actual:
            religion_actual[rel] = 0

for mar in marital_categories:
    col_name = f'count-{mar}'
    if col_name in marital_df.columns:
        marital_actual[mar] = marital_df[col_name].iloc[0]
    else:
        # Fallback to searching through columns
        for col in marital_df.columns:
            if col.endswith(mar) and col != 'geography code' and col != 'total':
                marital_actual[mar] = marital_df[col].iloc[0]
                break
        if mar not in marital_actual:
            marital_actual[mar] = 0

# If counts are still zero, try extracting from cross-tables
if sum(ethnicity_actual.values()) == 0:
    for eth in ethnicity_categories:
        total_eth = 0
        for col in ethnic_by_sex_by_age_df.columns:
            if f' {eth}' in col:
                total_eth += ethnic_by_sex_by_age_df[col].iloc[0]
        ethnicity_actual[eth] = total_eth

if sum(religion_actual.values()) == 0:
    for rel in religion_categories:
        total_rel = 0
        for col in religion_by_sex_by_age_df.columns:
            if f' {rel}' in col:
                total_rel += religion_by_sex_by_age_df[col].iloc[0]
        religion_actual[rel] = total_rel

if sum(marital_actual.values()) == 0:
    for mar in marital_categories:
        total_mar = 0
        for col in marital_by_sex_by_age_df.columns:
            if f' {mar}' in col:
                total_mar += marital_by_sex_by_age_df[col].iloc[0]
        marital_actual[mar] = total_mar

# Calculate predicted distributions
sex_pred_counts = dict(Counter(sex_pred_names))
age_pred_counts = dict(Counter(age_pred_names))
ethnicity_pred_counts = dict(Counter(ethnicity_pred_names))
religion_pred_counts = dict(Counter(religion_pred_names))
marital_pred_counts = dict(Counter(marital_pred_names))

# Normalize the actual distributions to match the total number of persons in predictions
# This ensures fair comparison of relative proportions
total_actual_sex = sum(sex_actual.values())
total_actual_age = sum(age_actual.values())
total_actual_ethnicity = sum(ethnicity_actual.values())
total_actual_religion = sum(religion_actual.values())
total_actual_marital = sum(marital_actual.values())
total_pred = num_persons

if total_actual_sex > 0:
    sex_actual = {k: v * total_pred / total_actual_sex for k, v in sex_actual.items()}
if total_actual_age > 0:
    age_actual = {k: v * total_pred / total_actual_age for k, v in age_actual.items()}
if total_actual_ethnicity > 0:
    ethnicity_actual = {k: v * total_pred / total_actual_ethnicity for k, v in ethnicity_actual.items()}
if total_actual_religion > 0:
    religion_actual = {k: v * total_pred / total_actual_religion for k, v in religion_actual.items()}
if total_actual_marital > 0:
    marital_actual = {k: v * total_pred / total_actual_marital for k, v in marital_actual.items()}

# Create attribute dictionaries for plotting
attribute_dicts = {
    'Sex': (sex_actual, sex_pred_counts),
    'Age': (age_actual, age_pred_counts),
    'Ethnicity': (ethnicity_actual, ethnicity_pred_counts),
    'Religion': (religion_actual, religion_pred_counts),
    'Marital Status': (marital_actual, marital_pred_counts)
}

categories_dict = {
    'Sex': sex_categories,
    'Age': age_groups,
    'Ethnicity': ethnicity_categories,
    'Religion': religion_categories,
    'Marital Status': marital_categories
}

# Plot individual attribute distributions
plotly_attribute_distributions(attribute_dicts, categories_dict, filter_zero_bars=True)

# Create crosstables for visualization
# Create actual crosstables from the three-way relationships
ethnic_sex_age_actual = pd.DataFrame(0, index=sex_categories, columns=age_groups)
religion_sex_age_actual = pd.DataFrame(0, index=sex_categories, columns=age_groups)
marital_sex_age_actual = pd.DataFrame(0, index=sex_categories, columns=age_groups)

# Extract the actual counts from the crosstable dataframes
for sex in sex_categories:
    for age in age_groups:
        # Sum up counts for each ethnicity for this sex-age combination
        for eth in ethnicity_categories:
            col_name = f'{sex} {age} {eth}'
            if col_name in ethnic_by_sex_by_age_df.columns:
                ethnic_sex_age_actual.loc[sex, age] += ethnic_by_sex_by_age_df[col_name].iloc[0]
        
        # Sum up counts for each religion for this sex-age combination
        for rel in religion_categories:
            col_name = f'{sex} {age} {rel}'
            if col_name in religion_by_sex_by_age_df.columns:
                religion_sex_age_actual.loc[sex, age] += religion_by_sex_by_age_df[col_name].iloc[0]
        
        # Sum up counts for each marital status for this sex-age combination
        for mar in marital_categories:
            col_name = f'{sex} {age} {mar}'
            if col_name in marital_by_sex_by_age_df.columns:
                marital_sex_age_actual.loc[sex, age] += marital_by_sex_by_age_df[col_name].iloc[0]

# Create predicted crosstables from our predictions
ethnic_sex_age_pred = pd.DataFrame(0, index=sex_categories, columns=age_groups)
religion_sex_age_pred = pd.DataFrame(0, index=sex_categories, columns=age_groups)
marital_sex_age_pred = pd.DataFrame(0, index=sex_categories, columns=age_groups)

# Fill the predicted crosstables based on our model predictions
for i in range(len(sex_pred_names)):
    sex = sex_pred_names[i]
    age = age_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    mar = marital_pred_names[i]
    
    ethnic_sex_age_pred.loc[sex, age] += 1
    religion_sex_age_pred.loc[sex, age] += 1
    marital_sex_age_pred.loc[sex, age] += 1

# Create crosstable dictionaries for plotting
actual_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_actual,
    'Religion_Sex_Age': religion_sex_age_actual,
    'Marital_Sex_Age': marital_sex_age_actual
}

predicted_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_pred,
    'Religion_Sex_Age': religion_sex_age_pred,
    'Marital_Sex_Age': marital_sex_age_pred
}

titles = [
    'Ethnicity by Sex by Age Distribution',
    'Religion by Sex by Age Distribution',
    'Marital Status by Sex by Age Distribution'
]

# Plot crosstable comparisons using the same function as in generateHouseholds.py
plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True)