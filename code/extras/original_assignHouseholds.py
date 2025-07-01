import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import pandas as pd

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
persons_file_path = os.path.join(current_dir, "./outputs/person_nodes.pt")
households_file_path = os.path.join(current_dir, "./outputs/household_nodes.pt")
hh_size_df = pd.read_csv(os.path.join(current_dir, '../../data/preprocessed-data/individual/HH_size.csv'))

# Define the Oxford areas
oxford_areas = ['E02005924']
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
household_nodes = torch.load(households_file_path)  # Example size: (num_households x 3)

# Convert to float for neural network compatibility
person_nodes = person_nodes.float()
household_nodes = household_nodes.float()

# Define the household composition categories and mapping
hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
hh_map = {category: i for i, category in enumerate(hh_compositions)}
reverse_hh_map = {v: k for k, v in hh_map.items()}  # Reverse mapping to decode

# Extract the household composition predictions
hh_pred = household_nodes[:, 0].long()

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
household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)
print("Done assigning household sizes")

# Step 2: Define the GNN model
class HouseholdAssignmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_households):
        super(HouseholdAssignmentGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)  # Added third layer
        self.fc = torch.nn.Linear(hidden_channels, num_households)

    def forward(self, x, edge_index):
        # GCN layers to process person nodes
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
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

# Step 3: Create the graph with more flexible edge construction (match on religion or ethnicity)
edge_index_file_path = os.path.join(current_dir, "edge_index.pt")
if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    print(f"Loaded edge index from {edge_index_file_path}")
else:
    edge_index = [[], []]  # Placeholder for edges
    cnt = 0
    for i in range(num_persons):
        if i % 10 == 0:
            print(i)
        for j in range(i + 1, num_persons):  # Avoid duplicate edges by starting at i + 1
            # Create an edge if either religion OR ethnicity matches
            if (person_nodes[i, religion_col_persons] == person_nodes[j, religion_col_persons] or
                person_nodes[i, ethnicity_col_persons] == person_nodes[j, ethnicity_col_persons]):
                edge_index[0].append(i)
                edge_index[1].append(j)
                # Since it's an undirected graph, add both directions
                edge_index[0].append(j)
                edge_index[1].append(i)
                cnt += 1
    print(f"Generated {cnt} edges")
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    torch.save(edge_index, edge_index_file_path)
    print(f"Edge index saved to {edge_index_file_path}")

# Step 4: Initialize the GNN model
in_channels = person_nodes.size(1)  # Assuming 5 characteristics per person
hidden_channels = 32  # Increased hidden channels
model = HouseholdAssignmentGNN(in_channels, hidden_channels, num_households)

# Move model to GPU
model = model.to(device)
print(f"Model moved to {device}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Adaptive LR

def compute_loss(assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0):
    # 1. Household size mismatch loss (MSE)
    household_counts = assignments.sum(dim=0)  # Sum the soft assignments across households
    size_loss = F.mse_loss(household_counts.float(), household_sizes.float())  # MSE loss for household size

    # 2. Religion loss (MSE for soft probabilities)
    religion_col_persons, religion_col_households = 2, 2  # Assuming column 2 for religion
    person_religion = person_nodes[:, religion_col_persons].float()  # Target (ground truth) religion as a float tensor
    predicted_religion_scores = assignments @ household_nodes[:, religion_col_households].float()  # Predicted religion (soft scores)
    religion_loss = F.mse_loss(predicted_religion_scores, person_religion)  # MSE loss for religion

    # 3. Ethnicity loss (MSE for soft probabilities)
    ethnicity_col_persons, ethnicity_col_households = 3, 1  # Assuming column 3 for ethnicity
    person_ethnicity = person_nodes[:, ethnicity_col_persons].float()  # Target (ground truth) ethnicity as a float tensor
    predicted_ethnicity_scores = assignments @ household_nodes[:, ethnicity_col_households].float()  # Predicted ethnicity (soft scores)
    ethnicity_loss = F.mse_loss(predicted_ethnicity_scores, person_ethnicity)  # MSE loss for ethnicity

    # Combine the losses with weights
    total_loss = size_loss + (religion_loss_weight * religion_loss) + (ethnicity_loss_weight * ethnicity_loss)

    return total_loss, size_loss, religion_loss, ethnicity_loss


# Compliance Accuracy Function (unchanged)
def calculate_individual_compliance_accuracy(assignments, person_nodes, household_nodes):
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1

    total_people = assignments.size(0)
    
    correct_religion_assignments = 0
    correct_ethnicity_assignments = 0

    # Loop over each person and their assigned household
    for person_idx, household_idx in enumerate(assignments):
        household_idx = household_idx.item()  # Get the household assignment for the person

        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]

        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]

        # Check if the person's religion matches the household's religion
        if person_religion == household_religion:
            correct_religion_assignments += 1

        # Check if the person's ethnicity matches the household's ethnicity
        if person_ethnicity == household_ethnicity:
            correct_ethnicity_assignments += 1

    religion_compliance = correct_religion_assignments / total_people
    ethnicity_compliance = correct_ethnicity_assignments / total_people

    return religion_compliance, ethnicity_compliance

# Household Size Accuracy Function (unchanged)
def calculate_size_distribution_accuracy(assignments, household_sizes):
    # Step 1: Calculate the predicted sizes (how many people in each household)
    predicted_counts = torch.zeros_like(household_sizes)  # Start with zeros for each household
    for household_idx in assignments:
        predicted_counts[household_idx] += 1  # Increment for each assignment
    
    # Step 2: Clamp both predicted and actual sizes to a maximum of 8
    predicted_counts_clamped = torch.clamp(predicted_counts, min=1, max=8)
    household_sizes_clamped = torch.clamp(household_sizes, min=1, max=8)

    # Step 3: Calculate bincount of the clamped predicted and actual sizes
    max_size = 8  # Since we clamped everything above size 8, the max size is now 8
    predicted_distribution = torch.bincount(predicted_counts_clamped, minlength=max_size).float()
    actual_distribution = torch.bincount(household_sizes_clamped, minlength=max_size).float()

    # Step 4: Calculate accuracy for each size
    accuracies = torch.min(predicted_distribution, actual_distribution) / (actual_distribution + 1e-6)  # Avoid division by 0
    overall_accuracy = accuracies.mean().item()  # Average accuracy across all household sizes

    return overall_accuracy

# Step 6: Training loop with combined loss for religion, ethnicity, and household size
epochs = 100
tau = 1.0  # Start with a higher tau
religion_loss_weight = 1.0  # Initial weight for religion loss
ethnicity_loss_weight = 1.0  # Initial weight for ethnicity loss

# Move training tensors to GPU
person_nodes_gpu = person_nodes.to(device)
household_nodes_gpu = household_nodes.to(device)
edge_index_gpu = edge_index.to(device)
household_sizes_gpu = household_sizes.to(device)
print(f"Training tensors moved to {device}")

for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass (all operations now on GPU)
    logits = model(person_nodes_gpu, edge_index_gpu)  # Shape: (num_persons, num_households)

    # Apply Gumbel-Softmax to get differentiable assignments
    assignments = gumbel_softmax(logits, tau=tau, hard=False)  # Shape: (num_persons, num_households)

    # Calculate the combined loss
    total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
        assignments, household_sizes_gpu, person_nodes_gpu, household_nodes_gpu, religion_loss_weight, ethnicity_loss_weight
    )
    
    # Backward pass
    total_loss.backward()

    # Clip gradients to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    # scheduler.step(total_loss)  # Adjust learning rate based on loss

    # Anneal the temperature for Gumbel-Softmax more slowly
    tau = max(0.5, tau * 0.995)

    # Step 7: Final assignments after training (move back to CPU for accuracy calculations)
    final_assignments = torch.argmax(assignments, dim=1).cpu()  # Get final discrete assignments

    # Calculate religion and ethnicity compliance accuracies (using CPU tensors)
    religion_compliance, ethnicity_compliance = calculate_individual_compliance_accuracy(
        final_assignments,       
        person_nodes,            # Use original CPU tensors for accuracy calculation
        household_nodes          
    )

    # Calculate household size distribution accuracy (using CPU tensors)
    household_size_accuracy = calculate_size_distribution_accuracy(final_assignments, household_sizes)

    # Print the loss and accuracies for the epoch
    print(f'Epoch {epoch}, Total Loss: {total_loss.item()}')
    print(f"Household size loss: {size_loss.item()}")
    print(f"Religion loss: {religion_loss.item()}")
    print(f"Ethnicity loss: {ethnicity_loss.item()}")
    print(f"Religion compliance accuracy: {religion_compliance * 100:.2f}%")
    print(f"Ethnicity compliance accuracy: {ethnicity_compliance * 100:.2f}%")
    print(f"Household size distribution accuracy: {household_size_accuracy * 100:.2f}%")
    
    # Optional: Clear GPU cache periodically to prevent memory issues
    if (epoch + 1) % 10 == 0:
        torch.cuda.empty_cache()