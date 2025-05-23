import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import pandas as pd

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
hh_map = {category: i for i, category in enumerate(hh_compositions)}
reverse_hh_map = {v: k for k, v in hh_map.items()}  # Reverse mapping to decode

# Extract the household composition predictions
# hh_pred = household_nodes[:, 0].long()
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
# household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)
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
edge_index_file_path = os.path.join(current_dir, "edge_index.pt")
if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    edge_index = edge_index.to(device)  # Move to the correct device
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
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    torch.save(edge_index, edge_index_file_path)
    print(f"Edge index saved to {edge_index_file_path}")

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

    total_loss = size_loss +  religion_loss +  ethnicity_loss
    return total_loss, size_loss, religion_loss, ethnicity_loss

# Step 4: Hyperparameter tuning setup
# learning_rates = [0.001, 0.0001, 0.0005]  # Define a range of learning rates
# hidden_dims = [64, 128, 256]  # Define a range of hidden dimensions
learning_rates = [0.001]  # Define a range of learning rates
hidden_dims = [64]  # Define a range of hidden dimensions
best_loss = float('inf')  # Initialize best loss to infinity
best_params = {}  # Store the best hyperparameters

# Function to perform training with given hyperparameters
def train_model(learning_rate, hidden_channels):
    model = HouseholdAssignmentGNN(in_channels=person_nodes.size(1), hidden_channels=hidden_channels, num_households=household_sizes.size(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tau = 1.0

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(person_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=tau, hard=False)

        total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
            assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0
        )
        total_loss.backward()
        optimizer.step()
        tau = max(0.5, tau * 0.995)

    return total_loss.item()

# Perform grid search over hyperparameters
for lr in learning_rates:
    for hidden_dim in hidden_dims:
        print(f"Training with learning rate {lr} and hidden dimension {hidden_dim}")
        final_loss = train_model(learning_rate=lr, hidden_channels=hidden_dim)
        print(f"Final loss: {final_loss}")

        # Track the best performing hyperparameters
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = {'learning_rate': lr, 'hidden_channels': hidden_dim}

# Output the best hyperparameters
print(f"Best hyperparameters: {best_params} with final loss {best_loss}")