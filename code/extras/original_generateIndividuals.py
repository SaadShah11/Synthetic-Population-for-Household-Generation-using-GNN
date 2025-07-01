import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
from torch.nn import CrossEntropyLoss
import random

# Device selection: Use MPS (Metal Performance Shaders) for Mac M1 GPU, or fallback to CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
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
age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Age_5yrs.csv'))
sex_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Sex.csv'))
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Ethnic.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Religion.csv'))
marital_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/Marital.csv'))
ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/ethnic_by_sex_by_age_modified.csv'))
religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/religion_by_sex_by_age.csv'))
marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/crosstables/marital_by_sex_by_age.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']

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
sex_categories = ['M', 'F']
ethnicity_categories = ['W0', 'M0', 'A0', 'B0', 'O0']
religion_categories = ['C','B','H','J','M','S','OR','NR','NS']
marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']

# Encode the categories to indices
age_map = {category: i for i, category in enumerate(age_groups)}
sex_map = {category: i for i, category in enumerate(sex_categories)}
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
marital_map = {category: i for i, category in enumerate(marital_categories)}

# Total number of persons from the total column
num_persons = int(age_df['total'].sum())

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
    def __init__(self, in_channels, hidden_channels, mlp_hidden_dim, out_channels_age, out_channels_sex, out_channels_ethnicity, out_channels_religion, out_channels_marital):
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
        self.dropout = torch.nn.Dropout(0.5)
        
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
# learning_rates = [0.001, 0.0005, 0.0001]
# hidden_channel_options = [64, 128, 256]
learning_rates = [0.001]
hidden_channel_options = [64]
mlp_hidden_dim = 128
num_epochs = 2500

# Results storage
results = []

# Define a function to train the model
def train_model(lr, hidden_channels, num_epochs, data, targets):
    # Initialize model, optimizer, and loss functions
    model = EnhancedGNNModelWithMLP(
        in_channels=node_features.size(1),
        hidden_channels=hidden_channels,
        mlp_hidden_dim=mlp_hidden_dim,
        out_channels_age=21,
        out_channels_sex=2,
        out_channels_ethnicity=5,
        out_channels_religion=9,
        out_channels_marital=6
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print metrics every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluate accuracy after training
    with torch.no_grad():
        age_out, sex_out, ethnicity_out, religion_out, marital_out = model(data)
        age_pred = age_out[:num_persons].argmax(dim=1)
        sex_pred = sex_out[:num_persons].argmax(dim=1)
        ethnicity_pred = ethnicity_out[:num_persons].argmax(dim=1)
        religion_pred = religion_out[:num_persons].argmax(dim=1)
        marital_pred = marital_out[:num_persons].argmax(dim=1)

        # Calculate net accuracy across all tasks
        net_accuracy = 0
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            pred_3 = out[targets[i][0][2]].argmax(dim=1)
            task_net_accuracy = ((pred_1 == targets[i][1][0]) & (pred_2 == targets[i][1][1]) & (pred_3 == targets[i][1][2])).sum().item() / num_persons
            net_accuracy += task_net_accuracy

        # Return the final loss, averaged net accuracy, model, and predictions
        predictions = {
            'age_pred': age_pred,
            'sex_pred': sex_pred, 
            'ethnicity_pred': ethnicity_pred,
            'religion_pred': religion_pred,
            'marital_pred': marital_pred
        }
        return loss.item(), net_accuracy / len(targets), model, predictions

# Run the grid search over hyperparameters
best_model = None
best_predictions = None
best_accuracy = 0

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Train the model for the current combination of hyperparameters
        final_loss, avg_accuracy, model, predictions = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            'average_accuracy': avg_accuracy
        })

        # Keep track of best model
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_model = model
            best_predictions = predictions

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}, Average Accuracy: {avg_accuracy}")

# After all runs, display results
results_df = pd.DataFrame(results)
print("Hyperparameter tuning results:")
print(results_df)

# Save the results to a CSV for future reference
output_path = os.path.join(current_dir, 'hyperparameter_tuning_results.csv')
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Save outputs in PyTorch format
outputs_dir = os.path.join(current_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

# Save the best trained model
if best_model is not None:
    model_path = os.path.join(outputs_dir, 'best_individual_model.pt')
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model state dict saved to {model_path}")

# Save predictions
if best_predictions is not None:
    predictions_path = os.path.join(outputs_dir, 'individual_predictions.pt')
    torch.save(best_predictions, predictions_path)
    print(f"Predictions saved to {predictions_path}")
    
    # Also save as structured tensor for assignHouseholds.py
    # Expected format: [age, sex, religion, ethnicity, marital] (5 columns)
    # Where religion is at index 2 and ethnicity is at index 3
    person_nodes_tensor = torch.stack([
        best_predictions['age_pred'],      # Column 0: age
        best_predictions['sex_pred'],      # Column 1: sex  
        best_predictions['religion_pred'], # Column 2: religion
        best_predictions['ethnicity_pred'], # Column 3: ethnicity
        best_predictions['marital_pred']   # Column 4: marital
    ], dim=1)
    
    person_nodes_path = os.path.join(outputs_dir, 'person_nodes.pt')
    torch.save(person_nodes_tensor, person_nodes_path)
    print(f"Person nodes tensor saved to {person_nodes_path}")

# Save data and other relevant tensors
data_artifacts = {
    'data': data,
    'targets': targets,
    'node_features': node_features,
    'edge_index': edge_index,
    'age_groups': age_groups,
    'sex_categories': sex_categories,
    'ethnicity_categories': ethnicity_categories,
    'religion_categories': religion_categories,
    'marital_categories': marital_categories,
    'age_map': age_map,
    'sex_map': sex_map,
    'ethnicity_map': ethnicity_map,
    'religion_map': religion_map,
    'marital_map': marital_map,
    'num_persons': num_persons
}
artifacts_path = os.path.join(outputs_dir, 'individual_data_artifacts.pt')
torch.save(data_artifacts, artifacts_path)
print(f"Data artifacts saved to {artifacts_path}")