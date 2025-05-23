import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import pandas as pd
import time
from datetime import timedelta

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
persons_file_path = os.path.join(current_dir, "./outputs/person_nodes.pt")
households_file_path = os.path.join(current_dir, "./outputs/household_nodes.pt")
# hh_size_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/HH_size.csv'))
hh_size_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_size.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
household_nodes = torch.load(households_file_path)  # Example size: (num_households x 3)

# Move tensors to the selected device
person_nodes = person_nodes.to(device)
household_nodes = household_nodes.to(device)

# Define the household composition categories and mapping
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
# hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-nC', '1FM-nA','1FC-0C','1FC-nC','1FC-nA','1FL-nA','1FL-nC','1H-nS','1H-nE','1H-nA', '1H-nC']
hh_map = {category: i for i, category in enumerate(hh_compositions)}
reverse_hh_map = {v: k for k, v in hh_map.items()}  # Reverse mapping to decode

# Extract the household composition predictions
hh_pred = household_nodes[:, 1].long()

# Flattening size and weight lists
values_size_org = [k for k in hh_size_df.columns if k not in ['geography code', 'total']]
weights_size_org = hh_size_df.iloc[0, 2:].tolist()  # Assuming first row, and skipping the first two columns

household_size_dist = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k != '1'}
values_size, weights_size = zip(*household_size_dist.items())

household_size_dist_na = {k: v for k, v in zip(hh_size_df.columns[2:], hh_size_df.iloc[0, 2:]) if k not in ['1', '2']}
values_size_na, weights_size_na = zip(*household_size_dist_na.items())

# Define the size assignment function based on household composition
fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C'}
# fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
# three_or_more_hh = {'1FM-nC', '1FM-nA', '1FC-nC', '1FC-nA'}
# two_or_more_hh = {'1FL-nC', '1FL-nA', '1H-nC'}

def fit_household_size(composition):
    if composition in fixed_hh:
        return fixed_hh[composition]
    elif composition in three_or_more_hh:
        return int(random.choices(values_size_na, weights=weights_size_na)[0].replace('8+', '8'))
    elif composition in two_or_more_hh:
        return int(random.choices(values_size, weights=weights_size)[0].replace('8+', '8'))
    else:
        return int(random.choices(values_size_org, weights=weights_size_org)[0].replace('8+', '8'))

# Assign sizes to each household based on its composition
household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long).to(device)
print("Done assigning household sizes")

# Step 2: Define the GNN model
class HouseholdAssignmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_households):
        super(HouseholdAssignmentGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)  # Added third layer
        self.fc = torch.nn.Linear(hidden_channels, num_households)

    def forward(self, x, edge_index):
        # GCN layers to process person nodes
        x = self.conv1(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()  # Added third GNN layer
        # Fully connected layer to output logits for each household
        out = self.fc(x)
        return out  # Output shape: (num_persons, num_households)

# Define Gumbel-Softmax
def gumbel_softmax(logits, tau=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)

    if hard:
        # Straight-through trick: take the index of the max value, but keep the gradient.
        y_hard = torch.zeros_like(logits).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

# Step 3: Create the graph
num_persons = person_nodes.size(0)
num_households = household_sizes.size(0)

# Define the columns for religion and ethnicity 
religion_col_persons, religion_col_households = 2, 2
ethnicity_col_persons, ethnicity_col_households = 3, 1

# Create the graph with more flexible edge construction (match on religion or ethnicity)
edge_index_file_path = os.path.join(current_dir, "outputs/edge_index.pt")
if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    edge_index = edge_index.to(device)  # Move to the correct device
    print(f"Loaded edge index from {edge_index_file_path}")
else:
    print(f"Creating new edge index and saving to {edge_index_file_path}")
    print("num_persons:", num_persons)
    
    # Extract religion and ethnicity vectors
    religion_vector = person_nodes[:, religion_col_persons].cpu()
    ethnicity_vector = person_nodes[:, ethnicity_col_persons].cpu()
    
    # Set to control maximum number of nodes to process (for very large datasets)
    # You can adjust this parameter based on your available memory
    max_batch_size = 1000  
    
    # Option to sample a subset of nodes for edge creation (speeds up computation)
    use_sampling = True if num_persons > 10000 else False
    sample_ratio = 0.2  # Adjust this value to control the fraction of nodes to sample
    
    if use_sampling:
        sample_size = int(num_persons * sample_ratio)
        sampled_indices = torch.randperm(num_persons)[:sample_size]
        print(f"Using sampling with {sample_size} nodes out of {num_persons}")
    else:
        sampled_indices = torch.arange(num_persons)
    
    edge_index = [[], []]
    cnt = 0
    
    # Process in batches to reduce memory usage
    for start_idx in range(0, len(sampled_indices), max_batch_size):
        batch_indices = sampled_indices[start_idx:start_idx + max_batch_size]
        print(f"Processing batch {start_idx//max_batch_size + 1}/{(len(sampled_indices) + max_batch_size - 1)//max_batch_size}")
        
        # Extract batch data
        batch_religions = religion_vector[batch_indices]
        batch_ethnicities = ethnicity_vector[batch_indices]
        
        # For each node in the batch
        for i, node_idx in enumerate(batch_indices):
            i_religion = batch_religions[i]
            i_ethnicity = batch_ethnicities[i]
            
            # Create a boolean mask of matching nodes (with either same religion OR same ethnicity)
            # Using vectorized operations instead of loop
            religion_matches = (religion_vector == i_religion)
            ethnicity_matches = (ethnicity_vector == i_ethnicity)
            matches = torch.logical_or(religion_matches, ethnicity_matches)
            
            # Exclude self-connection and get indices of matching nodes
            matches[node_idx] = False  # Remove self-match
            match_indices = torch.where(matches)[0]
            
            # Add edges for matches
            for j in match_indices:
                # Only add edge if j > node_idx to avoid duplicates
                if j > node_idx:
                    edge_index[0].append(node_idx.item())
                    edge_index[1].append(j.item())
                    # Since it's an undirected graph, add both directions
                    edge_index[0].append(j.item())
                    edge_index[1].append(node_idx.item())
                    cnt += 1
            
            # Print progress every 100 nodes
            if i % 100 == 0:
                print(f"  Processed {i}/{len(batch_indices)} nodes in current batch, found {cnt} edges so far")
    
    print(f"Generated {cnt} edges")
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    torch.save(edge_index.cpu(), edge_index_file_path)  # Save to CPU for storage
    print(f"Edge index saved to {edge_index_file_path}")

# Optional: Add code to limit edge count if the graph is too dense
# max_edges = 5000000  # Adjust based on your memory constraints
# if edge_index.shape[1] > max_edges:
#     print(f"Edge count ({edge_index.shape[1]}) exceeds maximum ({max_edges}). Randomly sampling edges.")
#     perm = torch.randperm(edge_index.shape[1])[:max_edges]
#     edge_index = edge_index[:, perm]
#     print(f"Reduced to {edge_index.shape[1]} edges")

# Compute loss function (as in the original code)
def compute_loss(assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0):
    household_counts = assignments.sum(dim=0)  # Sum the soft assignments across households
    size_loss = F.mse_loss(household_counts.float(), household_sizes.float())  # MSE loss for household size

    religion_col_persons, religion_col_households = 2, 2
    person_religion = person_nodes[:, religion_col_persons].float()  # Target (ground truth) religion as a float tensor
    predicted_religion_scores = assignments @ household_nodes[:, religion_col_households].float()  # Predicted religion (soft scores)
    religion_loss = F.mse_loss(predicted_religion_scores, person_religion)  # MSE loss for religion

    ethnicity_col_persons, ethnicity_col_households = 3, 1
    person_ethnicity = person_nodes[:, ethnicity_col_persons].float()  # Target (ground truth) ethnicity as a float tensor
    predicted_ethnicity_scores = assignments @ household_nodes[:, ethnicity_col_households].float()  # Predicted ethnicity (soft scores)
    ethnicity_loss = F.mse_loss(predicted_ethnicity_scores, person_ethnicity)  # MSE loss for ethnicity

    total_loss = size_loss + religion_loss_weight * religion_loss + ethnicity_loss_weight * ethnicity_loss
    return total_loss, size_loss, religion_loss, ethnicity_loss

# Function to compute accuracy metrics
def compute_accuracy(assignments, household_sizes, person_nodes, household_nodes):
    # Get hard assignments (one-hot)
    person_to_household = assignments.argmax(dim=1)
    
    # Compute household size accuracy
    assigned_counts = torch.bincount(person_to_household, minlength=num_households)
    size_matches = (assigned_counts == household_sizes).float()
    size_accuracy = size_matches.mean().item()
    
    # Compute religion matching accuracy
    person_religion = person_nodes[:, religion_col_persons]
    household_religion = household_nodes[:, religion_col_households]
    assigned_religion = household_religion[person_to_household]
    religion_matches = (person_religion == assigned_religion).float()
    religion_accuracy = religion_matches.mean().item()
    
    # Compute ethnicity matching accuracy
    person_ethnicity = person_nodes[:, ethnicity_col_persons]
    household_ethnicity = household_nodes[:, ethnicity_col_households]
    assigned_ethnicity = household_ethnicity[person_to_household]
    ethnicity_matches = (person_ethnicity == assigned_ethnicity).float()
    ethnicity_accuracy = ethnicity_matches.mean().item()
    
    # Overall accuracy (average of all metrics)
    overall_accuracy = (size_accuracy + religion_accuracy + ethnicity_accuracy) / 3
    
    return size_accuracy, religion_accuracy, ethnicity_accuracy, overall_accuracy

# Step 4: Hyperparameter tuning setup
# learning_rates = [0.001, 0.0001, 0.0005]  # Define a range of learning rates
learning_rates = [0.001]  # Define a range of learning rates
# hidden_dims = [64, 128, 256]  # Define a range of hidden dimensions
hidden_dims = [64]  # Define a range of hidden dimensions
num_epochs = 100  # Number of epochs to train

# Results storage
results = []
time_results = []

# Function to perform training with given hyperparameters
def train_model(learning_rate, hidden_channels, num_epochs):
    model = HouseholdAssignmentGNN(
        in_channels=person_nodes.size(1),
        hidden_channels=hidden_channels,
        num_households=household_sizes.size(0)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tau = 1.0
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(person_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=tau, hard=False)
        
        total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
            assignments, household_sizes, person_nodes, household_nodes, 
            religion_loss_weight=1.0, ethnicity_loss_weight=1.0
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Gradually decrease temperature
        tau = max(0.5, tau * 0.995)
        
        # Track losses
        loss_history.append({
            'epoch': epoch + 1,
            'total_loss': total_loss.item(),
            'size_loss': size_loss.item(),
            'religion_loss': religion_loss.item(),
            'ethnicity_loss': ethnicity_loss.item()
        })
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Total Loss: {total_loss.item():.4f}, '
                  f'Size Loss: {size_loss.item():.4f}, '
                  f'Religion Loss: {religion_loss.item():.4f}, '
                  f'Ethnicity Loss: {ethnicity_loss.item():.4f}')
    
    # Get final assignments
    with torch.no_grad():
        logits = model(person_nodes, edge_index)
        final_assignments = gumbel_softmax(logits, tau=0.5, hard=True)
        
        # Compute accuracy metrics on final assignments
        size_acc, religion_acc, ethnicity_acc, overall_acc = compute_accuracy(
            final_assignments, household_sizes, person_nodes, household_nodes
        )

    return (
        total_loss.item(),
        size_loss.item(),
        religion_loss.item(),
        ethnicity_loss.item(),
        size_acc,
        religion_acc,
        ethnicity_acc,
        overall_acc,
        final_assignments
    )

# Start total timing
total_start_time = time.time()

# Perform grid search over hyperparameters
for lr in learning_rates:
    for hidden_dim in hidden_dims:
        print(f"\nTraining with learning rate {lr} and hidden dimension {hidden_dim}")
        
        # Start timing for this combination
        start_time = time.time()
        
        total_loss, size_loss, religion_loss, ethnicity_loss, size_acc, religion_acc, ethnicity_acc, overall_acc, assignments = train_model(
            learning_rate=lr, 
            hidden_channels=hidden_dim,
            num_epochs=num_epochs
        )
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_dim,
            'total_loss': total_loss,
            'size_loss': size_loss,
            'religion_loss': religion_loss,
            'ethnicity_loss': ethnicity_loss,
            'size_accuracy': size_acc,
            'religion_accuracy': religion_acc,
            'ethnicity_accuracy': ethnicity_acc,
            'overall_accuracy': overall_acc,
            'training_time': train_time_str
        })
        
        # Store timing results separately
        time_results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_dim,
            'training_time': train_time_str
        })
        
        print(f"Finished training with lr={lr}, hidden_channels={hidden_dim}")
        print(f"Final Total Loss: {total_loss:.4f}")
        print(f"Final Size Loss: {size_loss:.4f}")
        print(f"Final Religion Loss: {religion_loss:.4f}")
        print(f"Final Ethnicity Loss: {ethnicity_loss:.4f}")
        print(f"Size Accuracy: {size_acc:.4f}")
        print(f"Religion Accuracy: {religion_acc:.4f}")
        print(f"Ethnicity Accuracy: {ethnicity_acc:.4f}")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Training time: {train_time_str}")

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total training time: {total_training_time_str}")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print("\nHyperparameter tuning results:")
print(results_df)

# Add total time as a summary row
total_row = pd.DataFrame([{
    'learning_rate': 'Total',
    'hidden_channels': '',
    'total_loss': '',
    'size_loss': '',
    'religion_loss': '',
    'ethnicity_loss': '',
    'size_accuracy': '',
    'religion_accuracy': '',
    'ethnicity_accuracy': '',
    'overall_accuracy': '',
    'training_time': total_training_time_str
}])
summary_df = pd.concat([results_df, total_row], ignore_index=True)

# Get the best hyperparameters (based on overall accuracy instead of just loss)
best_result = max(results, key=lambda x: x['overall_accuracy'])
print(f"\nBest hyperparameters: learning_rate={best_result['learning_rate']}, hidden_channels={best_result['hidden_channels']}")
print(f"Best overall accuracy: {best_result['overall_accuracy']:.4f}")

# Save the results to a CSV
output_dir = os.path.join(current_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'assignHouseholdsHPTuning_hyperparameter_tuning_results.csv')
summary_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Create detailed timing report as text file
timing_report_path = os.path.join(output_dir, 'assignHouseholdsHPTuning_training_time_results.txt')
with open(timing_report_path, 'w') as f:
    f.write("Hyperparameter Training Time Results\n")
    f.write("==================================\n\n")
    
    for i, result in enumerate(time_results):
        f.write(f"Combination {i+1}:\n")
        f.write(f"  Learning Rate: {result['learning_rate']}\n")
        f.write(f"  Hidden Channels: {result['hidden_channels']}\n")
        f.write(f"  Training Time: {result['training_time']}\n")
        f.write("\n")
    
    f.write("==================================\n")
    f.write(f"Total Training Time: {total_training_time_str}\n")
    f.write("==================================\n\n")
    
    f.write("Final Results Table:\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    
    # Add device information
    f.write(f"Training Device: {device}\n")
    f.write(f"Number of Epochs: {num_epochs}\n")

print(f"Training time results saved to {timing_report_path}")

# Train the model with the best hyperparameters to get the final assignments
print("\nTraining final model with best hyperparameters...")
model = HouseholdAssignmentGNN(
    in_channels=person_nodes.size(1),
    hidden_channels=best_result['hidden_channels'],
    num_households=household_sizes.size(0)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=best_result['learning_rate'])
tau = 1.0

for epoch in range(num_epochs * 2):  # Train for longer on the final model
    model.train()
    optimizer.zero_grad()
    
    logits = model(person_nodes, edge_index)
    assignments = gumbel_softmax(logits, tau=tau, hard=False)
    
    total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
        assignments, household_sizes, person_nodes, household_nodes, 
        religion_loss_weight=1.0, ethnicity_loss_weight=1.0
    )
    
    total_loss.backward()
    optimizer.step()
    
    tau = max(0.5, tau * 0.995)
    
    if (epoch + 1) % 20 == 0:
        print(f'Final Model - Epoch {epoch+1}/{num_epochs*2}, Total Loss: {total_loss.item():.4f}')

# Get final hard assignments
with torch.no_grad():
    logits = model(person_nodes, edge_index)
    final_assignments = gumbel_softmax(logits, tau=0.5, hard=True)
    
    # Calculate and print final accuracy metrics
    final_size_acc, final_religion_acc, final_ethnicity_acc, final_overall_acc = compute_accuracy(
        final_assignments, household_sizes, person_nodes, household_nodes
    )
    print(f"Final Model - Size Accuracy: {final_size_acc:.4f}")
    print(f"Final Model - Religion Accuracy: {final_religion_acc:.4f}")
    print(f"Final Model - Ethnicity Accuracy: {final_ethnicity_acc:.4f}")
    print(f"Final Model - Overall Accuracy: {final_overall_acc:.4f}")

# Create assignment tensor that maps persons to households
person_to_household = final_assignments.argmax(dim=1)
assignment_tensor = torch.zeros(num_persons, 2, device=device)
assignment_tensor[:, 0] = torch.arange(num_persons, device=device)  # Person ID
assignment_tensor[:, 1] = person_to_household  # Assigned Household ID

# Save the assignments
assignments_path = os.path.join(output_dir, 'assignHouseholdsHPTuning_person_to_household_assignments.pt')
torch.save(assignment_tensor.cpu(), assignments_path)
print(f"Person to household assignments saved to {assignments_path}")

# Save final model metrics
final_metrics = {
    'learning_rate': best_result['learning_rate'],
    'hidden_channels': best_result['hidden_channels'],
    'size_accuracy': final_size_acc,
    'religion_accuracy': final_religion_acc,
    'ethnicity_accuracy': final_ethnicity_acc,
    'overall_accuracy': final_overall_acc
}
metrics_df = pd.DataFrame([final_metrics])
metrics_path = os.path.join(output_dir, 'assignHouseholdsHPTuning_final_model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Final model metrics saved to {metrics_path}")
