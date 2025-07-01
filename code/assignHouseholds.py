import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import timedelta
import gc

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for household assignment using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# GPU Memory Management Functions
def print_gpu_memory_info(device, message=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        print("GPU memory cleared and reset")

def safe_delete_tensor(tensor):
    """Safely delete a tensor and free GPU memory"""
    if tensor is not None:
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def safe_delete_model(model):
    """Safely delete a model and free GPU memory"""
    if model is not None:
        # Move model to CPU first to free GPU memory
        if hasattr(model, 'cpu'):
            model = model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def monitor_memory_usage(device, message="", threshold_gb=8.0):
    """Monitor GPU memory usage and warn if approaching limit"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        
        # Warn if memory usage is high
        if allocated > threshold_gb:
            print(f"WARNING: High GPU memory usage detected ({allocated:.2f}GB). Consider reducing batch size or model complexity.")
            return True
        return False
    return False

def emergency_memory_cleanup():
    """Emergency memory cleanup when memory is critically low"""
    print("Performing emergency memory cleanup...")
    
    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("Emergency memory cleanup completed")

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Assignment Hyperparameter Tuning for area: {selected_area_code}")

# Set print options to display all elements of the tensor
torch.set_printoptions(edgeitems=torch.inf)

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    monitor_memory_usage(device, "at startup")

# Step 1: Load the tensors and household size data
current_dir = os.path.dirname(os.path.abspath(__file__))
# persons_file_path = os.path.join(current_dir, "./outputs/person_nodes.pt")
# households_file_path = os.path.join(current_dir, "./outputs/household_nodes.pt")
persons_file_path = os.path.join(current_dir, f"./outputs/individuals_{selected_area_code}/person_nodes.pt")
households_file_path = os.path.join(current_dir, f"./outputs/households_{selected_area_code}/household_nodes.pt")
hh_size_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_size.csv'))

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")
hh_size_df = hh_size_df[hh_size_df['geography code'].isin(oxford_areas)]

# Load the tensors from the files
person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
household_nodes = torch.load(households_file_path)  # Example size: (num_households x 3)

# Convert to float for neural network compatibility
person_nodes = person_nodes.float()
household_nodes = household_nodes.float()

# Move tensors to GPU
person_nodes = person_nodes.to(device)
household_nodes = household_nodes.to(device)
print(f"Moved person_nodes and household_nodes to {device}")

# Define the household composition categories and mapping
hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-2C', '1FM-nA', '1FC-0C', '1FC-2C', '1FC-nA', '1FL-nA', '1FL-2C', '1H-nS', '1H-nE', '1H-nA', '1H-2C']
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
# fixed_hh = {"1PE": 1, "1PA": 1, "1FM-0C": 2, "1FC-0C": 2}
# three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
# two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C'}

fixed_hh = {"1PE": 1, "1PA": 1, "1FE": 2, "1FM-0C": 2, "1FC-0C": 2}
three_or_more_hh = {'1FM-2C', '1FM-nA', '1FC-2C', '1FC-nA'}
two_or_more_hh = {'1FL-2C', '1FL-nA', '1H-2C', '1H-nS', '1H-nE', '1H-nA'}

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
household_sizes = household_sizes.to(device)
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
        y_hard = torch.zeros_like(logits, device=logits.device).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

# Step 3: Create the graph
num_persons = person_nodes.size(0)
num_households = household_sizes.size(0)

# Define the columns for religion and ethnicity 
# Based on actual tensor structures from generation scripts:
# person_nodes: [age, sex, religion, ethnicity, marital] (5 columns)
# household_nodes: [household_composition, ethnicity, religion] (3 columns)
religion_col_persons, religion_col_households = 2, 2
ethnicity_col_persons, ethnicity_col_households = 3, 1

# Create the graph with more flexible edge construction (match on religion or ethnicity)
# edge_index_file_path = os.path.join(current_dir, "output" , "edge_index.pt")
# edge_index_file_path = "./outputs/edge_index.pt"
edge_index_file_path = os.path.join(current_dir, f"./outputs/assignment_hp_tuning_{selected_area_code}/edge_index.pt")

# Create output directory for assignment results
output_dir = os.path.join(current_dir, 'outputs', f'assignment_hp_tuning_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(edge_index_file_path):
    edge_index = torch.load(edge_index_file_path)
    print(f"Loaded edge index from {edge_index_file_path}")
else:
    print("Creating edge index using optimized GPU operations...")
    
    # Use context manager for memory optimization during edge creation
    with torch.no_grad():  # Disable gradient computation for memory efficiency
        # Extract religion and ethnicity columns for efficient comparison
        person_religion = person_nodes[:, religion_col_persons]
        person_ethnicity = person_nodes[:, ethnicity_col_persons]
        
        # Create matrices for pairwise comparison using broadcasting
        # Shape: (num_persons, num_persons) - True where persons have same religion/ethnicity
        religion_match = person_religion.unsqueeze(1) == person_religion.unsqueeze(0)
        ethnicity_match = person_ethnicity.unsqueeze(1) == person_ethnicity.unsqueeze(0)
        
        # Combine matches: True if either religion OR ethnicity matches
        matches = religion_match | ethnicity_match
        
        # Create upper triangular mask to avoid duplicate edges (i < j)
        upper_tri_mask = torch.triu(torch.ones(num_persons, num_persons, device=device, dtype=torch.bool), diagonal=1)
        
        # Apply mask to only get upper triangular matches
        final_matches = matches & upper_tri_mask
        
        # Get indices where matches occur
        edge_sources, edge_targets = torch.where(final_matches)
        
        # Create bidirectional edges (undirected graph)
        edge_index = torch.stack([
            torch.cat([edge_sources, edge_targets]),  # Source nodes
            torch.cat([edge_targets, edge_sources])   # Target nodes
        ], dim=0)
        
        # Count unique edges (divide by 2 since we count each edge twice)
        cnt = edge_sources.size(0)
        print(f"Generated {cnt} edges using GPU optimization")
        
        # Clear intermediate tensors to free memory
        safe_delete_tensor(person_religion)
        safe_delete_tensor(person_ethnicity)
        safe_delete_tensor(religion_match)
        safe_delete_tensor(ethnicity_match)
        safe_delete_tensor(matches)
        safe_delete_tensor(upper_tri_mask)
        safe_delete_tensor(final_matches)
        safe_delete_tensor(edge_sources)
        safe_delete_tensor(edge_targets)
        
        # Move to CPU for saving
        edge_index_cpu = edge_index.cpu()
        torch.save(edge_index_cpu, edge_index_file_path)
        print(f"Edge index saved to {edge_index_file_path}")
        
        # Keep edge_index on GPU for further processing
        # edge_index remains on device

# Move edge index to GPU (if not already there)
if not edge_index.is_cuda and device.type == 'cuda':
    edge_index = edge_index.to(device)
    print(f"Moved edge_index to {device}")
else:
    print(f"Edge index already on {device}")

monitor_memory_usage(device, "after edge index creation")

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
num_epochs = 200  # Number of training epochs
learning_rates = [0.001]  # Expanded range of learning rates
hidden_dims = [64]  # Expanded range of hidden dimensions
# learning_rates = [0.001, 0.0001, 0.0005]  # Define a range of learning rates
# hidden_dims = [64, 128, 256]  # Define a range of hidden dimensions
best_loss = float('inf')  # Initialize best loss to infinity
best_params = {}  # Store the best hyperparameters

# Store all results for saving
hp_results = []
detailed_results = []  # Store detailed results for each hyperparameter combination
convergence_results = []  # Store convergence data for each combination

# Global best model tracking (similar to generateIndividuals.py)
best_model_info = {
    'model_state': None,
    'loss': float('inf'),
    'accuracy': 0,
    'assignments': None,
    'lr': None,
    'hidden_channels': None,
    'convergence_data': None,
    'detailed_accuracies': None,
    'epoch_numbers': None,
    'religion_accuracies': None,
    'ethnicity_accuracies': None
}

# Plotting functions
def plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir):
    """Plot assignment errors similar to assignment_model2.py"""
    
    # Calculate size errors
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in final_assignments:
        predicted_counts[household_idx] += 1
    
    size_errors = torch.abs(predicted_counts - household_sizes).sum().item()
    
    # Calculate religion and ethnicity errors
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    
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
    # plt.show()
    print(f"Assignment errors plot saved to: {error_plot_path}")

def plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, output_dir):
    """Plot accuracy over epochs with religion and ethnicity graphs side by side"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    # plt.show()
    print(f"Accuracy over epochs plot saved to: {accuracy_plot_path}")

# Household Size Accuracy Function (consistent with original script)
def calculate_size_distribution_accuracy(assignments, household_sizes):
    """
    Calculate household size distribution accuracy by comparing predicted vs expected size distributions.
    Uses the same method as the original assignHouseholds script for consistency.
    """
    # Step 1: Calculate the predicted sizes (how many people in each household)
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
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

# Compliance Accuracy Function
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

# Combined Accuracy Function
def calculate_all_accuracies(assignments, person_nodes, household_nodes, household_sizes):
    """
    Calculate all accuracies: religion, ethnicity, and household size.
    Returns individual accuracies and overall average accuracy.
    """
    # Calculate religion and ethnicity accuracies
    religion_compliance, ethnicity_compliance = calculate_individual_compliance_accuracy(
        assignments, person_nodes, household_nodes
    )
    
    # Calculate household size distribution accuracy (consistent with original script)
    size_distribution_accuracy = calculate_size_distribution_accuracy(
        assignments, household_sizes
    )
    
    # Calculate overall average accuracy
    overall_accuracy = (religion_compliance + ethnicity_compliance + size_distribution_accuracy) / 3.0
    
    return {
        'religion_compliance': religion_compliance,
        'ethnicity_compliance': ethnicity_compliance, 
        'size_distribution_accuracy': size_distribution_accuracy,
        'overall_accuracy': overall_accuracy
    }

# Function to perform training with given hyperparameters
def train_model(learning_rate, hidden_channels, return_detailed_results=False):
    print(f"    Starting training with LR={learning_rate}, Hidden={hidden_channels}")
    
    # Clear GPU memory before starting new training
    clear_gpu_memory()
    monitor_memory_usage(device, "before model creation")
    
    model = HouseholdAssignmentGNN(in_channels=person_nodes.size(1), hidden_channels=hidden_channels, num_households=household_sizes.size(0))
    model = model.to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tau = 1.0

    monitor_memory_usage(device, "after model creation")

    # Track accuracies over epochs and convergence data
    religion_accuracies = []
    ethnicity_accuracies = []
    epoch_numbers = []
    
    # Track convergence data for all training runs
    convergence_data = {
        'epochs': [],
        'losses': [],
        'size_losses': [],
        'religion_losses': [],
        'ethnicity_losses': [],
        'religion_accuracies': [],
        'ethnicity_accuracies': [],
        'size_distribution_accuracies': [],
        'overall_accuracies': [],
        'cumulative_time_seconds': [],
        'epoch_time_seconds': [],
        'tau_values': []
    }
    
    # Start timing for epoch-wise tracking
    training_start_time = time.time()
    best_epoch_loss = float('inf')
    best_epoch_state = None

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        logits = model(person_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=tau, hard=False)

        total_loss, size_loss, religion_loss, ethnicity_loss = compute_loss(
            assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0
        )
        total_loss.backward()
        
        # Clip gradients to avoid exploding gradients - Check if makes any change
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        tau = max(0.5, tau * 0.995)
        
        # Calculate all accuracies for this epoch
        final_assignments = torch.argmax(assignments, dim=1)
        accuracies = calculate_all_accuracies(final_assignments, person_nodes, household_nodes, household_sizes)
        
        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time = epoch_end_time - training_start_time
        
        # Store convergence data for every epoch
        convergence_data['epochs'].append(epoch + 1)
        convergence_data['losses'].append(total_loss.item())
        convergence_data['size_losses'].append(size_loss.item())
        convergence_data['religion_losses'].append(religion_loss.item())
        convergence_data['ethnicity_losses'].append(ethnicity_loss.item())
        convergence_data['religion_accuracies'].append(accuracies['religion_compliance'] * 100)
        convergence_data['ethnicity_accuracies'].append(accuracies['ethnicity_compliance'] * 100)
        convergence_data['size_distribution_accuracies'].append(accuracies['size_distribution_accuracy'] * 100)
        convergence_data['overall_accuracies'].append(accuracies['overall_accuracy'] * 100)
        convergence_data['epoch_time_seconds'].append(epoch_duration)
        convergence_data['cumulative_time_seconds'].append(cumulative_time)
        convergence_data['tau_values'].append(tau)
        
        # Track for detailed results
        epoch_numbers.append(epoch + 1)
        religion_accuracies.append(accuracies['religion_compliance'] * 100)
        ethnicity_accuracies.append(accuracies['ethnicity_compliance'] * 100)
        
        # Store best epoch state
        if total_loss.item() < best_epoch_loss:
            best_epoch_loss = total_loss.item()
            best_epoch_state = model.state_dict().copy()
        
        # Clear intermediate tensors to free memory
        safe_delete_tensor(logits)
        safe_delete_tensor(assignments)
        safe_delete_tensor(final_assignments)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\r    Epoch {epoch+1:3d}/{num_epochs} | Total Loss: {total_loss.item():.6f} | Size Loss: {size_loss.item():.6f} | Religion Loss: {religion_loss.item():.6f} | Ethnicity Loss: {ethnicity_loss.item():.6f} | Religion Acc: {accuracies['religion_compliance']*100:.2f}% | Ethnicity Acc: {accuracies['ethnicity_compliance']*100:.2f}% | Size Acc: {accuracies['size_distribution_accuracy']*100:.2f}% | Overall Acc: {accuracies['overall_accuracy']*100:.2f}% | Tau: {tau:.3f}", end="", flush=True)
            
            # Monitor memory usage every 10 epochs
            monitor_memory_usage(device, f"at epoch {epoch+1}")

    print()  # New line after training completes
    print(f"    Training completed. Final loss: {total_loss.item():.6f}")

    # Load best epoch state for final evaluation
    model.load_state_dict(best_epoch_state)
    
    # Get final assignments and accuracies using best model
    with torch.no_grad():
        logits = model(person_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=0.5, hard=True)
        final_assignments = torch.argmax(assignments, dim=1)
        final_accuracies = calculate_all_accuracies(final_assignments, person_nodes, household_nodes, household_sizes)

    # Update global best model info if this model performs better
    global best_model_info
    if final_accuracies['overall_accuracy'] > best_model_info['accuracy'] or (final_accuracies['overall_accuracy'] == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
        best_model_info.update({
            'model_state': best_epoch_state,
            'loss': best_epoch_loss,
            'accuracy': final_accuracies['overall_accuracy'],
            'assignments': final_assignments.clone(),
            'lr': learning_rate,
            'hidden_channels': hidden_channels,
            'convergence_data': convergence_data,
            'detailed_accuracies': final_accuracies,
            'epoch_numbers': epoch_numbers.copy(),
            'religion_accuracies': religion_accuracies.copy(),
            'ethnicity_accuracies': ethnicity_accuracies.copy()
        })

    # Clear model and intermediate tensors from GPU memory before returning
    safe_delete_tensor(logits)
    safe_delete_tensor(assignments)
    safe_delete_model(model)
    clear_gpu_memory()
    
    monitor_memory_usage(device, "after model cleanup")
    
    if return_detailed_results:
        return best_epoch_loss, final_assignments, epoch_numbers, religion_accuracies, ethnicity_accuracies, convergence_data, final_accuracies
    else:
        return best_epoch_loss, convergence_data, final_accuracies

# Perform grid search over hyperparameters
total_start_time = time.time()

# Clear GPU memory before starting hyperparameter tuning
clear_gpu_memory()
monitor_memory_usage(device, "before hyperparameter tuning")

try:
    for idx, lr in enumerate(learning_rates):
        for jdx, hidden_dim in enumerate(hidden_dims):
            try:
                combination_start_time = time.time()
                print(f"Training with learning rate {lr} and hidden dimension {hidden_dim} ({idx*len(hidden_dims)+jdx+1}/{len(learning_rates)*len(hidden_dims)})")
                
                # Print GPU memory before training
                monitor_memory_usage(device, "before training combination")
                
                final_loss, convergence_data, final_accuracies = train_model(learning_rate=lr, hidden_channels=hidden_dim)
                
                combination_end_time = time.time()
                combination_training_time = combination_end_time - combination_start_time
                combination_training_time_str = str(timedelta(seconds=int(combination_training_time)))
                
                print(f"Final loss: {final_loss:.6f}")
                print(f"Final Religion Compliance: {final_accuracies['religion_compliance']*100:.2f}%")
                print(f"Final Ethnicity Compliance: {final_accuracies['ethnicity_compliance']*100:.2f}%")
                print(f"Final Size Distribution Accuracy: {final_accuracies['size_distribution_accuracy']*100:.2f}%")
                print(f"Final Overall Accuracy: {final_accuracies['overall_accuracy']*100:.2f}%")
                print(f"Training time: {combination_training_time_str}")
                
                # Store basic results for saving
                hp_results.append({
                    'learning_rate': lr,
                    'hidden_channels': hidden_dim,
                    'final_loss': final_loss,
                    'religion_compliance': final_accuracies['religion_compliance'],
                    'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                    'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                    'overall_accuracy': final_accuracies['overall_accuracy'],
                    'training_time': combination_training_time_str
                })
                
                # Store detailed results
                detailed_results.append({
                    'learning_rate': lr,
                    'hidden_channels': hidden_dim,
                    'final_loss': final_loss,
                    'religion_compliance': final_accuracies['religion_compliance'],
                    'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                    'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                    'overall_accuracy': final_accuracies['overall_accuracy'],
                    'religion_accuracy_percent': final_accuracies['religion_compliance'] * 100,
                    'ethnicity_accuracy_percent': final_accuracies['ethnicity_compliance'] * 100,
                    'size_distribution_accuracy_percent': final_accuracies['size_distribution_accuracy'] * 100,
                    'overall_accuracy_percent': final_accuracies['overall_accuracy'] * 100,
                    'training_time_seconds': combination_training_time,
                    'training_time_str': combination_training_time_str,
                    'area_code': selected_area_code,
                    'num_persons': num_persons,
                    'num_households': household_sizes.size(0),
                    'num_epochs': num_epochs
                })
                
                # Store convergence data with hyperparameter info
                convergence_data_with_hp = convergence_data.copy()
                convergence_data_with_hp['learning_rate'] = [lr] * len(convergence_data['epochs'])
                convergence_data_with_hp['hidden_channels'] = [hidden_dim] * len(convergence_data['epochs'])
                convergence_data_with_hp['combination_id'] = [f"lr_{lr}_hc_{hidden_dim}"] * len(convergence_data['epochs'])
                convergence_results.append(convergence_data_with_hp)
                
                # Print GPU memory after training
                monitor_memory_usage(device, "after training combination")

                # Track the best performing hyperparameters (using overall accuracy as primary metric)
                if final_accuracies['overall_accuracy'] > best_params.get('overall_accuracy', 0) or (final_accuracies['overall_accuracy'] == best_params.get('overall_accuracy', 0) and final_loss < best_loss):
                    best_loss = final_loss
                    best_params = {
                        'learning_rate': lr, 
                        'hidden_channels': hidden_dim,
                        'religion_compliance': final_accuracies['religion_compliance'],
                        'ethnicity_compliance': final_accuracies['ethnicity_compliance'],
                        'size_distribution_accuracy': final_accuracies['size_distribution_accuracy'],
                        'overall_accuracy': final_accuracies['overall_accuracy']
                    }
                
                # Clear memory between combinations
                clear_gpu_memory()
                print("-" * 50)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GPU out of memory error for combination LR={lr}, Hidden={hidden_dim}")
                    print(f"Error: {e}")
                    print("Clearing GPU memory and continuing with next combination...")
                    emergency_memory_cleanup()
                    continue
                else:
                    print(f"Runtime error for combination LR={lr}, Hidden={hidden_dim}: {e}")
                    clear_gpu_memory()
                    continue
            except Exception as e:
                print(f"Unexpected error for combination LR={lr}, Hidden={hidden_dim}: {e}")
                clear_gpu_memory()
                continue

except Exception as e:
    print(f"Critical error during hyperparameter tuning: {e}")
    print("Performing emergency cleanup...")
    emergency_memory_cleanup()
    raise

# Calculate total training time
total_end_time = time.time()
total_training_time = total_end_time - total_start_time
total_training_time_str = str(timedelta(seconds=int(total_training_time)))
print(f"Total hyperparameter tuning time: {total_training_time_str}")

# Output the best hyperparameters
print(f"Best hyperparameters: {best_params} with final loss {best_loss}")

# Save hyperparameter tuning results
print(f"\nSaving hyperparameter tuning results to: {output_dir}")

# Save basic results as CSV
hp_results_df = pd.DataFrame(hp_results)
hp_results_path = os.path.join(output_dir, 'hp_tuning_results.csv')
hp_results_df.to_csv(hp_results_path, index=False)
print(f"Basic results saved: {hp_results_path}")

# Save detailed results as CSV
detailed_results_df = pd.DataFrame(detailed_results)
detailed_results_path = os.path.join(output_dir, 'detailed_hp_results.csv')
detailed_results_df.to_csv(detailed_results_path, index=False)
print(f"Detailed results saved: {detailed_results_path}")

# Save convergence data for all combinations
if convergence_results:
    # Combine all convergence data into one DataFrame
    all_convergence_data = []
    for conv_data in convergence_results:
        # Convert to DataFrame and add to list
        conv_df = pd.DataFrame(conv_data)
        all_convergence_data.append(conv_df)
    
    # Concatenate all convergence data
    combined_convergence_df = pd.concat(all_convergence_data, ignore_index=True)
    convergence_path = os.path.join(output_dir, 'all_combinations_convergence_data.csv')
    combined_convergence_df.to_csv(convergence_path, index=False)
    print(f"Convergence data for all combinations saved: {convergence_path}")

# Save performance summary
performance_summary = {
    'area_code': selected_area_code,
    'num_persons': num_persons,
    'num_households': household_sizes.size(0),
    'total_combinations_tested': len(hp_results),
    'total_training_time_seconds': total_training_time,
    'total_training_time_str': total_training_time_str,
    'num_epochs_per_combination': num_epochs,
    'learning_rates_tested': learning_rates,
    'hidden_dims_tested': hidden_dims,
    'best_learning_rate': best_params['learning_rate'],
    'best_hidden_channels': best_params['hidden_channels'],
    'best_loss': best_loss,
    'best_religion_compliance': best_params['religion_compliance'],
    'best_ethnicity_compliance': best_params['ethnicity_compliance'],
    'best_size_distribution_accuracy': best_params['size_distribution_accuracy'],
    'best_overall_accuracy': best_params['overall_accuracy']
}

performance_summary_path = os.path.join(output_dir, 'performance_summary.json')
with open(performance_summary_path, 'w') as f:
    json.dump(performance_summary, f, indent=4)
print(f"Performance summary saved: {performance_summary_path}")

# Save best parameters as JSON (enhanced)
best_params_with_loss = best_params.copy()
best_params_with_loss['best_loss'] = best_loss
best_params_with_loss['total_combinations'] = len(hp_results)
best_params_with_loss['area_code'] = selected_area_code
best_params_with_loss['religion_accuracy_percent'] = best_params['religion_compliance'] * 100
best_params_with_loss['ethnicity_accuracy_percent'] = best_params['ethnicity_compliance'] * 100
best_params_with_loss['size_distribution_accuracy_percent'] = best_params['size_distribution_accuracy'] * 100
best_params_with_loss['overall_accuracy_percent'] = best_params['overall_accuracy'] * 100
best_params_with_loss['total_training_time'] = total_training_time_str

best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params_with_loss, f, indent=4)
print(f"Best parameters saved: {best_params_path}")

print(f"\nAll hyperparameter tuning results saved to: {output_dir}")

# Use saved best model information instead of retraining
print(f"\n{'='*60}")
print(f"USING BEST MODEL RESULTS FOR PLOTTING (NO RETRAINING)")
print(f"{'='*60}")

# Print best model information
print("\nBest Model Information:")
print(f"Learning Rate: {best_model_info['lr']}")
print(f"Hidden Channels: {best_model_info['hidden_channels']}")
print(f"Best Loss: {best_model_info['loss']:.6f}")
print(f"Best Overall Accuracy: {best_model_info['accuracy']:.4f}")

# Extract saved results from best model
final_assignments = best_model_info['assignments']
epoch_numbers = best_model_info['epoch_numbers']
religion_accuracies = best_model_info['religion_accuracies']
ethnicity_accuracies = best_model_info['ethnicity_accuracies']
best_convergence_data = best_model_info['convergence_data']
final_accuracies = best_model_info['detailed_accuracies']

print(f"\nUsing saved best model results (no retraining needed)")

# Generate plots using saved data
print("\nGenerating plots...")
plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir)
plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, output_dir)

# Save final assignment results
print(f"\nSaving final assignment results to {output_dir}")

# Save final assignments tensor
final_assignments_path = os.path.join(output_dir, 'final_assignments.pt')
torch.save(final_assignments.cpu(), final_assignments_path)

# Save convergence data from best model run
best_convergence_df = pd.DataFrame(best_convergence_data)
best_convergence_path = os.path.join(output_dir, 'best_model_convergence_data.csv')
best_convergence_df.to_csv(best_convergence_path, index=False)
print(f"Best model convergence data saved: {best_convergence_path}")

# Update best parameters with final results from best model run
best_params_with_loss['final_religion_compliance'] = final_accuracies['religion_compliance']
best_params_with_loss['final_ethnicity_compliance'] = final_accuracies['ethnicity_compliance']
best_params_with_loss['final_size_distribution_accuracy'] = final_accuracies['size_distribution_accuracy']
best_params_with_loss['final_overall_accuracy'] = final_accuracies['overall_accuracy']
best_params_with_loss['final_religion_accuracy_percent'] = final_accuracies['religion_compliance'] * 100
best_params_with_loss['final_ethnicity_accuracy_percent'] = final_accuracies['ethnicity_compliance'] * 100
best_params_with_loss['final_size_distribution_accuracy_percent'] = final_accuracies['size_distribution_accuracy'] * 100
best_params_with_loss['final_overall_accuracy_percent'] = final_accuracies['overall_accuracy'] * 100

# Re-save updated best parameters
with open(best_params_path, 'w') as f:
    json.dump(best_params_with_loss, f, indent=4)

print(f"\nFinal Results with Best Hyperparameters:")
print(f"  Learning Rate: {best_model_info['lr']}")
print(f"  Hidden Channels: {best_model_info['hidden_channels']}")
print(f"  Final Loss: {best_model_info['loss']:.6f}")
print(f"  Religion Compliance: {final_accuracies['religion_compliance'] * 100:.2f}%")
print(f"  Ethnicity Compliance: {final_accuracies['ethnicity_compliance'] * 100:.2f}%")
print(f"  Size Distribution Accuracy: {final_accuracies['size_distribution_accuracy'] * 100:.2f}%")
print(f"  Overall Accuracy: {final_accuracies['overall_accuracy'] * 100:.2f}%")
print(f"  Results and plots saved to: {output_dir}")

# Comprehensive final cleanup
print("\nPerforming final memory cleanup...")

# Clear all intermediate variables
safe_delete_tensor(final_assignments)
safe_delete_tensor(epoch_numbers)
safe_delete_tensor(religion_accuracies)
safe_delete_tensor(ethnicity_accuracies)

# Clear data tensors if they're no longer needed
# Note: We keep person_nodes, household_nodes, household_sizes, and edge_index 
# as they might be needed for future operations
monitor_memory_usage(device, "before final cleanup")

# Clear GPU cache and perform garbage collection
clear_gpu_memory()

# Print final memory status
monitor_memory_usage(device, "after final cleanup")

# Print peak memory usage if available
if torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"Peak GPU Memory Usage: Allocated: {peak_allocated:.2f}GB, Reserved: {peak_reserved:.2f}GB")

print("Hyperparameter tuning completed!")
print("GPU memory has been cleared and optimized.")