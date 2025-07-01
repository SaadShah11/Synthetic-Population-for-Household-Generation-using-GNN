import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Assign individuals to households using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Assignment for area: {selected_area_code}")

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Check for CUDA availability and set device
# Device selection with additional safety check
if torch.cuda.is_available():
    try:
        # Test CUDA functionality with a simple operation
        test_tensor = torch.tensor([1.0]).cuda()
        test_result = test_tensor + 1
        device = torch.device('cuda')
        print(f"CUDA test passed. Using device: {device}")
        del test_tensor, test_result  # Clean up
    except Exception as e:
        print(f"CUDA test failed: {e}. Falling back to CPU.")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print(f"CUDA not available. Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
persons_file_path = os.path.join(current_dir, f"./outputs/individuals_{selected_area_code}/person_nodes.pt")
households_file_path = os.path.join(current_dir, f"./outputs/households_{selected_area_code}/household_nodes.pt")
hh_size_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_size.csv'))

# Define the Oxford areas
# oxford_areas = ['E02005924']

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
# persons_file_path = os.path.join(current_dir, "./outputs/person_nodes.pt")
# households_file_path = os.path.join(current_dir, "./outputs/household_nodes.pt")
person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
household_nodes = torch.load(households_file_path)  # Example size: (num_households x 3)

# Print tensor information before moving to GPU
print(f"Person nodes shape: {person_nodes.shape}")
print(f"Household nodes shape: {household_nodes.shape}")
print(f"Number of persons: {person_nodes.shape[0]}")
print(f"Number of households: {household_nodes.shape[0]}")

# Move tensors to GPU
person_nodes = person_nodes.to(device)
household_nodes = household_nodes.to(device)
print(f"Moved person_nodes and household_nodes to {device}")

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
household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)

# Move household_sizes to device with error handling
try:
    household_sizes = household_sizes.to(device)
    print(f"✓ Moved household_sizes to {device}")
except RuntimeError as e:
    print(f"✗ Failed to move household_sizes to CUDA: {e}")
    household_sizes = household_sizes.to('cpu')
    print(f"  Keeping household_sizes on CPU")

print(f"Done assigning household sizes. Shape: {household_sizes.shape}")
print(f"Household sizes range: [{household_sizes.min().item()}, {household_sizes.max().item()}]")
print(f"Total expected persons from household sizes: {household_sizes.sum().item()}")

# POPULATION BALANCING: Critical fix for household size loss
expected_persons = household_sizes.sum().item()
actual_persons = person_nodes.size(0)
person_difference = actual_persons - expected_persons

print(f"\n🔍 POPULATION BALANCE CHECK:")
print(f"  Expected persons: {expected_persons}")
print(f"  Actual persons: {actual_persons}")
print(f"  Difference: {person_difference}")

if person_difference != 0:
    print(f"⚠️ MISMATCH DETECTED: Adjusting household sizes...")
    
    # Strategy: Adjust household sizes proportionally
    if person_difference > 0:
        # Too many persons - increase some household sizes
        print(f"  Adding {person_difference} persons to households")
        adjustable_households = (household_sizes < 8).nonzero(as_tuple=True)[0]  # Households that can grow
        
        if len(adjustable_households) > 0:
            # Distribute extra persons across adjustable households
            additions_per_hh = person_difference // len(adjustable_households)
            remainder = person_difference % len(adjustable_households)
            
            # Add base amount to all adjustable households
            household_sizes[adjustable_households] += additions_per_hh
            
            # Add remainder to first few households
            if remainder > 0:
                household_sizes[adjustable_households[:remainder]] += 1
        else:
            # If no households can grow, increase the largest ones (overflow handling)
            largest_households = household_sizes.argsort(descending=True)[:person_difference]
            household_sizes[largest_households] += 1
            
    else:
        # Too few persons - decrease some household sizes
        print(f"  Removing {abs(person_difference)} persons from households")
        reducible_households = (household_sizes > 1).nonzero(as_tuple=True)[0]  # Households that can shrink
        
        if len(reducible_households) >= abs(person_difference):
            # Reduce household sizes by 1 for the needed number of households
            households_to_reduce = reducible_households[:abs(person_difference)]
            household_sizes[households_to_reduce] -= 1
        else:
            # More complex reduction needed
            reductions_per_hh = abs(person_difference) // len(reducible_households)
            remainder = abs(person_difference) % len(reducible_households)
            
            # Reduce all reducible households
            reduction_amount = torch.clamp(household_sizes[reducible_households] - 1, min=0)
            max_reduction = torch.min(reduction_amount, torch.tensor(reductions_per_hh, device=device))
            household_sizes[reducible_households] -= max_reduction
            
            # Handle remainder
            if remainder > 0:
                additional_reducible = (household_sizes[reducible_households[:remainder]] > 1)
                household_sizes[reducible_households[:remainder][additional_reducible]] -= 1

    # Verify the adjustment
    new_expected_persons = household_sizes.sum().item()
    print(f"✅ BALANCED: New expected persons: {new_expected_persons}")
    print(f"  Final household sizes range: [{household_sizes.min().item()}, {household_sizes.max().item()}]")
    
    if new_expected_persons != actual_persons:
        print(f"⚠️ WARNING: Perfect balance not achieved. Difference: {actual_persons - new_expected_persons}")
else:
    print(f"✅ PERFECT BALANCE: No adjustment needed")

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
        y_hard = torch.zeros_like(logits, device=logits.device).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

# Step 3: Create the graph
num_persons = person_nodes.size(0)
num_households = household_sizes.size(0)

print(f"Actual number of persons: {num_persons}")
print(f"Number of households: {num_households}")
expected_total = household_sizes.sum().item()
print(f"Final validation - Expected persons: {expected_total}, Actual persons: {num_persons}")

# Final constraint validation
if expected_total != num_persons:
    print(f"🚨 CRITICAL ERROR: Population still not balanced after adjustment!")
    print(f"   This will cause large household size loss during training.")
    print(f"   Consider regenerating individuals or households with better alignment.")
    
    # Calculate the magnitude of the problem
    mismatch_percent = abs(expected_total - num_persons) / num_persons * 100
    print(f"   Mismatch magnitude: {mismatch_percent:.1f}%")
    
    if mismatch_percent > 10:
        print(f"   ⚠️ WARNING: Mismatch > 10% - expect very large size loss")
    elif mismatch_percent > 5:
        print(f"   ⚠️ WARNING: Mismatch > 5% - expect moderate size loss")
else:
    print(f"✅ PERFECT BALANCE CONFIRMED: Training should proceed with low size loss")

# Define the columns for religion and ethnicity 
# Corrected based on actual tensor structures:
# person_nodes: [ID, Age, Religion, Ethnicity, Marital, Sex]  
# household_nodes: [ID, HH_Composition, Ethnicity, Religion]
religion_col_persons, religion_col_households = 2, 3
ethnicity_col_persons, ethnicity_col_households = 3, 2

# Step 3: Create the graph with more flexible edge construction (match on religion or ethnicity)
# edge_index_file_path = os.path.join(current_dir, "outputs", "edge_index.pt")
# edge_index_file_path = "./outputs/edge_index.pt"
edge_index_file_path = os.path.join(current_dir, f"./outputs/assignment_{selected_area_code}/edge_index.pt")

# Create output directory for assignment results
output_dir = os.path.join(current_dir, 'outputs', f'assignment_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    print(f"Loaded edge index from {edge_index_file_path}")
    
    # Validate edge index against current number of persons
    max_edge_index = edge_index.max().item() if edge_index.numel() > 0 else -1
    print(f"Max edge index: {max_edge_index}, Number of persons: {num_persons}")
    
    if max_edge_index >= num_persons:
        print(f"WARNING: Edge index contains indices >= {num_persons}. Regenerating edge index...")
        # Force regeneration by removing the file
        os.remove(edge_index_file_path)
        edge_index = None
    else:
        print("Edge index validation passed.")
else:
    edge_index = None

if edge_index is None:
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
    
    # Validate the newly generated edge index
    if edge_index.numel() > 0:
        max_edge_index = edge_index.max().item()
        min_edge_index = edge_index.min().item()
        print(f"Edge index range: [{min_edge_index}, {max_edge_index}], Number of persons: {num_persons}")
        
        if max_edge_index >= num_persons or min_edge_index < 0:
            raise ValueError(f"Invalid edge index: range [{min_edge_index}, {max_edge_index}] is not valid for {num_persons} persons")
    
    torch.save(edge_index, edge_index_file_path)
    print(f"Edge index saved to {edge_index_file_path}")

# Final validation before moving to GPU
print(f"Final edge index shape: {edge_index.shape}")
print(f"Edge index data type: {edge_index.dtype}")
if edge_index.numel() > 0:
    print(f"Edge index min/max: [{edge_index.min().item()}, {edge_index.max().item()}]")

# Move edge index to GPU
edge_index = edge_index.to(device)
print(f"Moved edge_index to {device}")

# Step 4: Initialize the GNN model
in_channels = person_nodes.size(1)  # Assuming 5 characteristics per person
hidden_channels = 32  # Increased hidden channels
model = HouseholdAssignmentGNN(in_channels, hidden_channels, num_households)
model = model.to(device)  # Move model to GPU
print(f"Moved model to {device}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Adaptive LR

def compute_loss(assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0):
    # 1. Household size mismatch loss (Improved: Huber loss + normalization)
    household_counts = assignments.sum(dim=0)  # Sum the soft assignments across households
    
    # Use Huber loss instead of MSE to reduce impact of large errors
    size_errors = household_counts.float() - household_sizes.float()
    huber_delta = 1.0  # Threshold for switching from quadratic to linear
    size_loss = F.smooth_l1_loss(household_counts.float(), household_sizes.float(), reduction='mean')
    
    # Optional: Add relative error component for better scaling
    relative_errors = size_errors / (household_sizes.float() + 1e-8)  # Avoid division by zero
    relative_loss = relative_errors.abs().mean()
    
    # Combine absolute and relative losses
    size_loss = size_loss + 0.1 * relative_loss

    # 2. Religion loss (MSE for soft probabilities)
    religion_col_persons, religion_col_households = 2, 3  # Assuming column 2 for religion
    person_religion = person_nodes[:, religion_col_persons].float()  # Target (ground truth) religion as a float tensor
    predicted_religion_scores = assignments @ household_nodes[:, religion_col_households].float()  # Predicted religion (soft scores)
    religion_loss = F.mse_loss(predicted_religion_scores, person_religion)  # MSE loss for religion

    # 3. Ethnicity loss (MSE for soft probabilities)
    ethnicity_col_persons, ethnicity_col_households = 3, 2  # Assuming column 3 for ethnicity
    person_ethnicity = person_nodes[:, ethnicity_col_persons].float()  # Target (ground truth) ethnicity as a float tensor
    predicted_ethnicity_scores = assignments @ household_nodes[:, ethnicity_col_households].float()  # Predicted ethnicity (soft scores)
    ethnicity_loss = F.mse_loss(predicted_ethnicity_scores, person_ethnicity)  # MSE loss for ethnicity

    # Combine the losses with weights
    total_loss = size_loss + (religion_loss_weight * religion_loss) + (ethnicity_loss_weight * ethnicity_loss)

    return total_loss, size_loss, religion_loss, ethnicity_loss


# Compliance Accuracy Function (unchanged)
def calculate_individual_compliance_accuracy(assignments, person_nodes, household_nodes):
    religion_col_persons, religion_col_households = 2, 3
    ethnicity_col_persons, ethnicity_col_households = 3, 2

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
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)  # Start with zeros for each household
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

# Track accuracies over epochs for plotting
religion_accuracies = []
ethnicity_accuracies = []
size_accuracies = []
epoch_numbers = []

for epoch in range(epochs):
    optimizer.zero_grad()

    try:
        # Forward pass
        logits = model(person_nodes, edge_index)  # Shape: (num_persons, num_households)
    except RuntimeError as e:
        if "CUDA" in str(e) and device.type == 'cuda':
            print(f"CUDA error encountered: {e}")
            print("Switching to CPU and retrying...")
            
            # Move everything to CPU
            device = torch.device('cpu')
            person_nodes = person_nodes.cpu()
            household_nodes = household_nodes.cpu()
            household_sizes = household_sizes.cpu()
            edge_index = edge_index.cpu()
            model = model.cpu()
            
            # Recreate optimizer for CPU
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            print("Switched to CPU. Retrying forward pass...")
            logits = model(person_nodes, edge_index)
        else:
            raise e

    # Apply Gumbel-Softmax to get differentiable assignments
    assignments = gumbel_softmax(logits, tau=tau, hard=False)  # Shape: (num_persons, num_households)

    # Calculate the combined loss
    total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
        assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight, ethnicity_loss_weight
    )
    
    # Backward pass
    total_loss.backward()

    # Clip gradients to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    # scheduler.step(total_loss)  # Adjust learning rate based on loss

    # Anneal the temperature for Gumbel-Softmax more slowly
    tau = max(0.5, tau * 0.995)

    # Step 7: Final assignments after training
    final_assignments = torch.argmax(assignments, dim=1)  # Get final discrete assignments

    # Calculate religion and ethnicity compliance accuracies
    religion_compliance, ethnicity_compliance = calculate_individual_compliance_accuracy(
        final_assignments,       
        person_nodes,            
        household_nodes          
    )

    # Calculate household size distribution accuracy
    household_size_accuracy = calculate_size_distribution_accuracy(final_assignments, household_sizes)

    # Track accuracies for plotting
    epoch_numbers.append(epoch + 1)
    religion_accuracies.append(religion_compliance * 100)
    ethnicity_accuracies.append(ethnicity_compliance * 100)
    size_accuracies.append(household_size_accuracy * 100)
    
    # Print the loss and accuracies for the epoch
    print(f'Epoch {epoch}, Total Loss: {total_loss.item()}')
    print(f"Household size loss: {size_loss.item()}")
    print(f"Religion loss: {religion_loss.item()}")
    print(f"Ethnicity loss: {ethnicity_loss.item()}")
    print(f"Religion compliance accuracy: {religion_compliance * 100:.2f}%")
    print(f"Ethnicity compliance accuracy: {ethnicity_compliance * 100:.2f}%")
    print(f"Household size distribution accuracy: {household_size_accuracy * 100:.2f}%")
    
    # Print GPU memory usage if using CUDA
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    print("-" * 50)

# Clear GPU cache at the end
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU cache cleared.")

# Plotting functions
def plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes):
    """Plot assignment errors similar to assignment_model2.py"""
    
    # Calculate size errors
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in final_assignments:
        predicted_counts[household_idx] += 1
    
    size_errors = torch.abs(predicted_counts - household_sizes).sum().item()
    
    # Calculate religion and ethnicity errors
    religion_col_persons, religion_col_households = 2, 3
    ethnicity_col_persons, ethnicity_col_households = 3, 2
    
    religion_errors = 0
    ethnicity_errors = 0
    
    for person_idx, household_idx in enumerate(final_assignments):
        household_idx = household_idx.item()
        
        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]
        
        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]
        
        if person_religion != household_religion:
            religion_errors += 1
        if person_ethnicity != household_ethnicity:
            ethnicity_errors += 1
    
    # Create bar graph
    plt.figure(figsize=(12, 6))
    
    categories = ['Size Errors', 'Religion Errors', 'Ethnicity Errors']
    error_counts = [size_errors, religion_errors, ethnicity_errors]
    colors = ['lightcoral', 'skyblue', 'lightgreen']
    
    bars = plt.bar(categories, error_counts, color=colors)
    
    # Add value labels on top of bars
    for bar, count in zip(bars, error_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Error Type')
    plt.ylabel('Number of Errors')
    plt.title('Assignment Errors by Type')
    plt.tight_layout()
    
    # Save plot
    error_plot_path = os.path.join(output_dir, 'assignment_errors.png')
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Assignment errors plot saved to: {error_plot_path}")

def plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies):
    """Plot accuracy over epochs similar to the provided image"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot (a) Religion
    bars1 = ax1.bar(epoch_numbers[::10], religion_accuracies[::10], color='steelblue', alpha=0.7, width=8)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Percentage of correctly assigned persons (%)')
    ax1.set_title('(a) Religion')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on top of bars (every 10th epoch)
    for i, (epoch, acc) in enumerate(zip(epoch_numbers[::10], religion_accuracies[::10])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax1.text(epoch, acc + 1, f'{acc:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot (b) Ethnicity
    bars2 = ax2.bar(epoch_numbers[::10], ethnicity_accuracies[::10], color='steelblue', alpha=0.7, width=8)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Percentage of correctly assigned persons (%)')
    ax2.set_title('(b) Ethnicity')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on top of bars (every 10th epoch)
    for i, (epoch, acc) in enumerate(zip(epoch_numbers[::10], ethnicity_accuracies[::10])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax2.text(epoch, acc + 1, f'{acc:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_over_epochs.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy over epochs plot saved to: {accuracy_plot_path}")

# Generate plots
print("\nGenerating plots...")
plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes)
plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies)

# Save final assignment results
print(f"\nSaving final assignment results to {output_dir}")

# Save final assignments tensor
final_assignments_path = os.path.join(output_dir, 'final_assignments.pt')
torch.save(final_assignments.cpu(), final_assignments_path)

# Create assignment summary
assignment_summary = {
    'total_persons': num_persons,
    'total_households': num_households,
    'final_religion_compliance': religion_compliance,
    'final_ethnicity_compliance': ethnicity_compliance,
    'final_household_size_accuracy': household_size_accuracy,
    'final_total_loss': total_loss.item(),
    'final_size_loss': size_loss.item(),
    'final_religion_loss': religion_loss.item(),
    'final_ethnicity_loss': ethnicity_loss.item()
}

# Save assignment summary as JSON
import json
summary_path = os.path.join(output_dir, 'assignment_summary.json')
with open(summary_path, 'w') as f:
    json.dump(assignment_summary, f, indent=4)

print(f"Assignment completed successfully!")
print(f"Final Results:")
print(f"  Religion compliance: {religion_compliance * 100:.2f}%")
print(f"  Ethnicity compliance: {ethnicity_compliance * 100:.2f}%")
print(f"  Household size accuracy: {household_size_accuracy * 100:.2f}%")
print(f"  Results saved to: {output_dir}")