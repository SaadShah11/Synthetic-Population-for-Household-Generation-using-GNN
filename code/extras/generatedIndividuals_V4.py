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
import argparse

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic individuals using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Individual Generation for area: {selected_area_code}")

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
# age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Age_Simplified.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Marital.csv'))
# ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/EthnicityBySexByAge_Simplified.csv'))
# religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ReligionbySexbyAge_Simplified.csv'))
# marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/MaritalbySexbyAgeModified_Simplified.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/EthnicityBySexByAge.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ReligionbySexbyAge.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/MaritalbySexbyAgeModified.csv'))

# Define the Oxford areas
# oxford_areas = ['E02005924']
# oxford_areas = ['E02005923']
# oxford_areas = ['E02005925']

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")

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
# age_groups = ['kids', 'adults', 'elders']
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
        
        # Batch normalization
        self.batch_norm1 = GraphNorm(hidden_channels)
        self.batch_norm2 = GraphNorm(hidden_channels)
        self.batch_norm3 = GraphNorm(hidden_channels)
        self.batch_norm4 = GraphNorm(hidden_channels)
        
        # Dropout layer
        # self.dropout = torch.nn.Dropout(0.5)
        
        # MLP for each output attribute
        self.mlp_age = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_age)
        )
        
        self.mlp_sex = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_sex)
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
        
        self.mlp_marital = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden_dim, out_channels_marital)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Pass through GraphSAGE layers
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.batch_norm4(x)
        x = F.relu(x)
        # x = self.dropout(x)
        
        # Pass the node embeddings through the MLPs for final attribute predictions
        age_out = self.mlp_age(x)
        sex_out = self.mlp_sex(x)
        ethnicity_out = self.mlp_ethnicity(x)
        religion_out = self.mlp_religion(x)
        marital_out = self.mlp_marital(x)
        
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
# learning_rates = [0.001]
# hidden_channel_options = [64]
learning_rates = [0.001, 0.0005, 0.0001]
hidden_channel_options = [64, 128, 256]
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
    
    # Custom accuracy function for multi-task learning
    def calculate_task_accuracy(pred_1, pred_2, pred_3, target_1, target_2, target_3):
        pred_1_labels = pred_1.argmax(dim=1)
        pred_2_labels = pred_2.argmax(dim=1)
        pred_3_labels = pred_3.argmax(dim=1)
        correct = ((pred_1_labels == target_1) & (pred_2_labels == target_2) & (pred_3_labels == target_3)).float()
        accuracy = correct.mean().item()
        return accuracy
    
    # Track best epoch state
    best_epoch_loss = float('inf')
    best_epoch_state = None
    loss_data = {}
    accuracy_data = {}
    
    # Storage for tracking metrics across epochs
    epoch_accuracies = []

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
        epoch_task_accuracies = []
        
        # Calculate losses and accuracies for all target combinations
        for i in range(len(targets)):
            current_loss = custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
                targets[i][1][0], targets[i][1][1], targets[i][1][2]
            )
            loss += current_loss
            
            # Calculate accuracy for this task
            task_accuracy = calculate_task_accuracy(
                out[targets[i][0][0]], out[targets[i][0][1]], out[targets[i][0][2]],
                targets[i][1][0], targets[i][1][1], targets[i][1][2]
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
        final_task_accuracies = {}
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            
            # Calculate joint accuracy - only counts as correct if ALL THREE predictions match the targets
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & 
                                (pred_2 == targets[i][1][1]) & 
                                (pred_3 == targets[i][1][2])).sum().item() / num_persons
            
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
                'predictions': (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred),
                'lr': lr,
                'hidden_channels': hidden_channels
            })

        # Return the final loss, average accuracy across epochs, and final accuracies
        return best_epoch_loss, average_accuracy, final_accuracy, (sex_pred, age_pred, ethnicity_pred, religion_pred, marital_pred)

# Run the grid search over hyperparameters
total_start_time = time.time()
time_results = []

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
output_dir = os.path.join(current_dir, 'outputs', f'individuals_{selected_area_code}')
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

sex_pred_names = [sex_categories[i] for i in sex_pred.cpu().numpy()]
age_pred_names = [age_groups[i] for i in age_pred.cpu().numpy()]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred.cpu().numpy()]
religion_pred_names = [religion_categories[i] for i in religion_pred.cpu().numpy()]
marital_pred_names = [marital_categories[i] for i in marital_pred.cpu().numpy()]

# # Calculate predicted distributions
# sex_pred_counts = dict(Counter([sex_categories[i] for i in sex_pred.cpu().numpy()]))
# age_pred_counts = dict(Counter([age_groups[i] for i in age_pred.cpu().numpy()]))
# ethnicity_pred_counts = dict(Counter([ethnicity_categories[i] for i in ethnicity_pred.cpu().numpy()]))
# religion_pred_counts = dict(Counter([religion_categories[i] for i in religion_pred.cpu().numpy()]))
# marital_pred_counts = dict(Counter([marital_categories[i] for i in marital_pred.cpu().numpy()]))

# Calculate actual distributions
sex_actual = {}
age_actual = {}
ethnicity_actual = {}
religion_actual = {}
marital_actual = {}

# Extract counts from the original data frames
for sex in sex_categories:
    sex_actual[sex] = sex_df[sex].iloc[0]
    # col_name = f'count-{sex}'
    # if col_name in sex_df.columns:
    #     sex_actual[sex] = sex_df[col_name].iloc[0]
    # else:
    #     # Fallback to searching through columns
    #     for col in sex_df.columns:
    #         if col.endswith(sex) and col != 'geography code' and col != 'total':
    #             sex_actual[sex] = sex_df[col].iloc[0]
    #             break
    #     if sex not in sex_actual:
    #         sex_actual[sex] = 0

for age in age_groups:
    age_actual[age] = age_df[age].iloc[0]
    # col_name = f'count-{age}'
    # if col_name in age_df.columns:
    #     age_actual[age] = age_df[col_name].iloc[0]
    # else:
    #     # Fallback to searching through columns
    #     for col in age_df.columns:
    #         if col.endswith(age) and col != 'geography code' and col != 'total':
    #             age_actual[age] = age_df[col].iloc[0]
    #             break
    #     if age not in age_actual:
    #         age_actual[age] = 0

for eth in ethnicity_categories:
    ethnicity_actual[eth] = ethnicity_df[eth].iloc[0]
    # col_name = f'count-{eth}'
    # if col_name in ethnicity_df.columns:
    #     ethnicity_actual[eth] = ethnicity_df[col_name].iloc[0]
    # else:
    #     # Fallback to searching through columns
    #     for col in ethnicity_df.columns:
    #         if col.endswith(eth) and col != 'geography code' and col != 'total':
    #             ethnicity_actual[eth] = ethnicity_df[col].iloc[0]
    #             break
    #     if eth not in ethnicity_actual:
    #         ethnicity_actual[eth] = 0

# for ethnicity in ethnicity_categories:
#     ethnicity_actual[ethnicity] = ethnicity_df[ethnicity].iloc[0]

for rel in religion_categories:
    religion_actual[rel] = religion_df[rel].iloc[0]
    # col_name = f'count-{rel}'
    # if col_name in religion_df.columns:
    #     religion_actual[rel] = religion_df[col_name].iloc[0]
    # else:
    #     # Fallback to searching through columns
    #     for col in religion_df.columns:
    #         if col.endswith(rel) and col != 'geography code' and col != 'total':
    #             religion_actual[rel] = religion_df[col].iloc[0]
    #             break
    #     if rel not in religion_actual:
    #         religion_actual[rel] = 0

for mar in marital_categories:
    marital_actual[mar] = marital_df[mar].iloc[0]
    # col_name = f'count-{mar}'
    # if col_name in marital_df.columns:
    #     marital_actual[mar] = marital_df[col_name].iloc[0]
    # else:
    #     # Fallback to searching through columns
    #     for col in marital_df.columns:
    #         if col.endswith(mar) and col != 'geography code' and col != 'total':
    #             marital_actual[mar] = marital_df[col].iloc[0]
    #             break
    #     if mar not in marital_actual:
    #         marital_actual[mar] = 0

# for marital in marital_categories:
#     marital_actual[marital] = marital_df[marital].iloc[0]

# If counts are still zero, try extracting from cross-tables
# if sum(ethnicity_actual.values()) == 0:
#     for eth in ethnicity_categories:
#         total_eth = 0
#         for col in ethnic_by_sex_by_age_df.columns:
#             if f' {eth}' in col:
#                 total_eth += ethnic_by_sex_by_age_df[col].iloc[0]
#         ethnicity_actual[eth] = total_eth

# if sum(religion_actual.values()) == 0:
#     for rel in religion_categories:
#         total_rel = 0
#         for col in religion_by_sex_by_age_df.columns:
#             if f' {rel}' in col:
#                 total_rel += religion_by_sex_by_age_df[col].iloc[0]
#         religion_actual[rel] = total_rel

# if sum(marital_actual.values()) == 0:
#     for mar in marital_categories:
#         total_mar = 0
#         for col in marital_by_sex_by_age_df.columns:
#             if f' {mar}' in col:
#                 total_mar += marital_by_sex_by_age_df[col].iloc[0]
#         marital_actual[mar] = total_mar

# print("============ Actual Values =============")
# print("\nSex:")
# print(sex_actual)
# print("\nAge:")
# print(age_actual)
# print("\nEthnicity:")
# print(ethnicity_actual)
# print("\nReligion:")
# print(religion_actual)
# print("\nMarital:")
# print(marital_actual)

# Calculate predicted distributions
sex_pred_counts = dict(Counter(sex_pred_names))
age_pred_counts = dict(Counter(age_pred_names))
ethnicity_pred_counts = dict(Counter(ethnicity_pred_names))
religion_pred_counts = dict(Counter(religion_pred_names))
marital_pred_counts = dict(Counter(marital_pred_names))

# print("============ Predicted Values =============")
# print("\nSex:")
# print(sex_pred_counts)
# print("\nAge:")
# print(age_pred_counts)
# print("\nEthnicity:")
# print(ethnicity_pred_counts)
# print("\nReligion:")
# print(religion_pred_counts)
# print("\nMarital:")
# print(marital_pred_counts)

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

# Create combined age-sex column names
age_sex_combinations = [f"{age} {sex}" for age in age_groups for sex in sex_categories]

# Create actual crosstables with ethnicity/religion/marital as indices and age-sex combinations as columns
ethnic_sex_age_actual = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
religion_sex_age_actual = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
marital_sex_age_actual = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)

# Extract the actual counts from the crosstable dataframes
for sex in sex_categories:
    for age in age_groups:
        col_name = f"{age} {sex}"
        # Sum up counts for each ethnicity for this sex-age combination
        for eth in ethnicity_categories:
            original_col = f'{sex} {age} {eth}'
            if original_col in ethnic_by_sex_by_age_df.columns:
                ethnic_sex_age_actual.loc[eth, col_name] = ethnic_by_sex_by_age_df[original_col].iloc[0]
        
        # Sum up counts for each religion for this sex-age combination
        for rel in religion_categories:
            original_col = f'{sex} {age} {rel}'
            if original_col in religion_by_sex_by_age_df.columns:
                religion_sex_age_actual.loc[rel, col_name] = religion_by_sex_by_age_df[original_col].iloc[0]
        
        # Sum up counts for each marital status for this sex-age combination
        for mar in marital_categories:
            original_col = f'{sex} {age} {mar}'
            if original_col in marital_by_sex_by_age_df.columns:
                marital_sex_age_actual.loc[mar, col_name] = marital_by_sex_by_age_df[original_col].iloc[0]

# print("============ Actual Crosstables =============")
# print("\nEthnicity:")
# print(ethnic_sex_age_actual)
# print("\nReligion:")
# print(religion_sex_age_actual)
# print("\nMarital:")
# print(marital_sex_age_actual)

# Create predicted crosstables with the same structure
ethnic_sex_age_pred = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
religion_sex_age_pred = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
marital_sex_age_pred = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)

# Fill the predicted crosstables based on our model predictions
for i in range(len(sex_pred_names)):
    sex = sex_pred_names[i]
    age = age_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    mar = marital_pred_names[i]
    
    col_name = f"{age} {sex}"
    ethnic_sex_age_pred.loc[eth, col_name] += 1
    religion_sex_age_pred.loc[rel, col_name] += 1
    marital_sex_age_pred.loc[mar, col_name] += 1

# print("============ Predicted Crosstables =============")
# print("\nEthnicity:")
# print(ethnic_sex_age_pred)
# print("\nReligion:")
# print(religion_sex_age_pred)
# print("\nMarital:")
# print(marital_sex_age_pred)

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
        
        # Flatten the dataframes to create 1D arrays for bar charts
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
        
        # Calculate label step size based on number of bars
        # Show fewer labels if there are many bars
        num_bars = len(continuous_positions)
        if num_bars > 300:
            step_size = 3
        elif num_bars > 100:
            step_size = 2
        else:
            step_size = 1
            
        # Create visible labels array with appropriate step size
        visible_labels = []
        visible_positions = []
        for i, (pos, label) in enumerate(zip(continuous_positions, original_indices_labels)):
            if i % step_size == 0:
                visible_labels.append(str(label))
                visible_positions.append(pos)
            else:
                visible_labels.append("")
                visible_positions.append(pos)
        
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
            ticktext=visible_labels,
            tickvals=visible_positions,
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
    
    # Display the plot
    fig.show()

# Prepare data for visualization
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
# Create crosstable dictionaries for plotting
actual_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_actual,
    'Religion_Sex_Age': religion_sex_age_actual,
    'Marital_Sex_Age': marital_sex_age_actual
}

# print("============ Actual Crosstables DF =============")
# print(actual_dfs)

predicted_dfs = {
    'Ethnic_Sex_Age': ethnic_sex_age_pred,
    'Religion_Sex_Age': religion_sex_age_pred,
    'Marital_Sex_Age': marital_sex_age_pred
}

# print("============ Predicted Crosstables DF =============")
# print(predicted_dfs)

titles = [
    'Ethnicity x Sex x Age',
    'Religion x Sex x Age',
    'Marital Status x Sex x Age'
]

# Plot crosstable comparisons using the same function as in generateHouseholds.py
plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True)

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

# Plot radar chart comparisons
plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles)

# Create glossary tables for crosstable numeric indices
def create_crosstable_glossary_individuals(row_categories, col_categories, crosstable_name):
    """
    Creates a glossary mapping numeric indices to actual category labels for crosstables.
    
    Parameters:
    row_categories - List of row categories (e.g., ethnicity, religion, marital categories)
    col_categories - List of column categories (age-sex combinations)
    crosstable_name - Name of the crosstable for the CSV filename
    
    Returns:
    DataFrame with columns: Sequential_Index, Row_Category, Column_Category, Full_Label
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

# # Create age-sex combinations for column labels
# age_sex_combinations = [f"{age} {sex}" for age in age_groups for sex in sex_categories]

# # Create glossaries for all three crosstables
# print("\n" + "="*60)
# print("CREATING CROSSTABLE GLOSSARIES")
# print("="*60)

# # Glossary for Ethnicity by Sex by Age
# ethnicity_sex_age_glossary = create_crosstable_glossary_individuals(
#     ethnicity_categories, 
#     age_sex_combinations, 
#     'Ethnicity_by_Sex_by_Age'
# )

# print(f"\nEthnicity by Sex by Age Glossary:")
# print(f"Total combinations: {len(ethnicity_sex_age_glossary)}")
# print("Sample entries:")
# print(ethnicity_sex_age_glossary.head(10))

# # Glossary for Religion by Sex by Age
# religion_sex_age_glossary = create_crosstable_glossary_individuals(
#     religion_categories, 
#     age_sex_combinations, 
#     'Religion_by_Sex_by_Age'
# )

# print(f"\nReligion by Sex by Age Glossary:")
# print(f"Total combinations: {len(religion_sex_age_glossary)}")
# print("Sample entries:")
# print(religion_sex_age_glossary.head(10))

# # Glossary for Marital Status by Sex by Age
# marital_sex_age_glossary = create_crosstable_glossary_individuals(
#     marital_categories, 
#     age_sex_combinations, 
#     'Marital_Status_by_Sex_by_Age'
# )

# print(f"\nMarital Status by Sex by Age Glossary:")
# print(f"Total combinations: {len(marital_sex_age_glossary)}")
# print("Sample entries:")
# print(marital_sex_age_glossary.head(10))

# print(f"\nAll individual crosstable glossary files saved to: {output_dir}")