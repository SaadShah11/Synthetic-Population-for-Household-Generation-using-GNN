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

# Household size extraction function
def extract_household_sizes_from_tensor(household_nodes_tensor, device):
    """
    Extract household size categories from the generated household tensor.
    
    Args:
        household_nodes_tensor: Tensor with shape (num_households, 6)
                               [household_composition, ethnicity, religion, tenure, size, rooms]
        device: PyTorch device for tensor operations
    
    Returns:
        torch.Tensor: Household size category indices as tensor of shape (num_households,)
        bool: True if extraction successful, False if fallback needed
    """
    try:
        # Validate tensor structure
        if household_nodes_tensor.dim() != 2:
            print(f"Warning: Expected 2D tensor, got {household_nodes_tensor.dim()}D tensor")
            return None, False
            
        if household_nodes_tensor.size(1) < 5:  # Need at least 5 columns to access index 4
            print(f"Warning: Expected at least 5 columns, got {household_nodes_tensor.size(1)} columns")
            return None, False
        
        # Extract size category indices from column 4 (0-indexed)
        size_category_indices = household_nodes_tensor[:, 4].long()
        
        # Ensure indices are on the correct device
        if size_category_indices.device != device:
            size_category_indices = size_category_indices.to(device)
        
        # Validate that indices are within the expected range (0-3 for 4 categories)
        valid_indices = (size_category_indices >= 0) & (size_category_indices < 4)
        if not valid_indices.all():
            invalid_count = (~valid_indices).sum().item()
            print(f"Warning: {invalid_count} households have invalid size category indices")
            # Clamp invalid indices to valid range
            size_category_indices = torch.clamp(size_category_indices, 0, 3)
        
        print(f"Successfully extracted household size categories from tensor. Category distribution:")
        unique_categories, counts = torch.unique(size_category_indices, return_counts=True)
        size_categories = ['1', '2', '3', '4+']
        for cat_idx, count in zip(unique_categories.cpu().numpy(), counts.cpu().numpy()):
            category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
            print(f"  Category {cat_idx} ({category_name}): {count} households")
            
        return size_category_indices, True
        
    except Exception as e:
        print(f"Error extracting household sizes from tensor: {e}")
        return None, False

def validate_household_size_extraction(household_nodes_tensor, device):
    """
    Validate the household size category extraction functionality with sample data.
    
    Args:
        household_nodes_tensor: The loaded household tensor
        device: PyTorch device
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("\n=== Validating Household Size Category Extraction ===")
    
    try:
        # Test the extraction function
        extracted_categories, success = extract_household_sizes_from_tensor(household_nodes_tensor, device)
        
        if not success:
            print("Validation FAILED: Size category extraction was not successful")
            return False
        
        # Basic validation checks
        num_households = household_nodes_tensor.size(0)
        if extracted_categories.size(0) != num_households:
            print(f"Validation FAILED: Expected {num_households} categories, got {extracted_categories.size(0)}")
            return False
        
        # Check if categories are within valid range (0-3 for 4 categories)
        min_cat = extracted_categories.min().item()
        max_cat = extracted_categories.max().item()
        
        if min_cat < 0 or max_cat > 3:
            print(f"Validation WARNING: Household size categories outside expected range [0-3]: min={min_cat}, max={max_cat}")
        
        # Check for reasonable distribution
        unique_categories, counts = torch.unique(extracted_categories, return_counts=True)
        size_categories = ['1', '2', '3', '4+']
        print("Extracted size category distribution validation:")
        for cat_idx, count in zip(unique_categories.cpu().numpy(), counts.cpu().numpy()):
            percentage = (count / num_households) * 100
            category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
            print(f"  Category {cat_idx} ({category_name}): {count} households ({percentage:.1f}%)")
        
        print("Validation PASSED: Household size category extraction is working correctly")
        return True
        
    except Exception as e:
        print(f"Validation FAILED: Exception during validation: {e}")
        return False

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
try:
    person_nodes = torch.load(persons_file_path)  # Example size: (num_persons x 5)
    print(f"Loaded person_nodes with shape: {person_nodes.shape}")
except Exception as e:
    print(f"Error loading person nodes from {persons_file_path}: {e}")
    raise

try:
    household_nodes = torch.load(households_file_path)  # Expected size: (num_households x 6)
    print(f"Loaded household_nodes with shape: {household_nodes.shape}")
    
    # Validate household tensor structure for size extraction
    if household_nodes.dim() == 2 and household_nodes.size(1) >= 5:
        print(f"Household tensor structure is compatible for size extraction (has {household_nodes.size(1)} columns)")
    else:
        print(f"Warning: Household tensor structure may not support size extraction (shape: {household_nodes.shape})")
        
except Exception as e:
    print(f"Error loading household nodes from {households_file_path}: {e}")
    raise

# Convert to float for neural network compatibility
person_nodes = person_nodes.float()
household_nodes = household_nodes.float()

# Move tensors to GPU
person_nodes = person_nodes.to(device)
household_nodes = household_nodes.to(device)
print(f"Moved person_nodes and household_nodes to {device}")

# Define the household composition categories and mapping
# hh_compositions = ['1PE', '1PA', '1FE', '1FM-0C', '1FM-nC', '1FM-nA', '1FC-0C', '1FC-nC', '1FC-nA', '1FL-nC', '1FL-nA', '1H-nC', '1H-nS', '1H-nE', '1H-nA']
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
three_or_more_hh = {'1FM-nC', '1FM-nA', '1FC-nC', '1FC-nA'}
two_or_more_hh = {'1FL-nC', '1FL-nA', '1H-nC', '1H-nS', '1H-nE', '1H-nA'}

def fit_household_size(composition):
    if composition in fixed_hh:
        return fixed_hh[composition]
    elif composition in three_or_more_hh:
        return int(random.choices(values_size_na, weights=weights_size_na)[0].replace('8+', '8'))
    elif composition in two_or_more_hh:
        return int(random.choices(values_size, weights=weights_size)[0].replace('8+', '8'))
    else:
        return int(random.choices(values_size_org, weights=weights_size_org)[0].replace('8+', '8'))

# Validate household size category extraction functionality
validation_passed = validate_household_size_extraction(household_nodes, device)

# Try to extract household size categories from the generated household tensor
household_size_categories, extraction_successful = extract_household_sizes_from_tensor(household_nodes, device)

if not extraction_successful:
    print("Extraction Failed. Exiting...")
    exit()
    # print("Falling back to random household size assignment based on composition...")
    # # Fallback: Assign sizes to each household based on its composition (original logic)
    # household_sizes = torch.tensor([fit_household_size(reverse_hh_map[hh_pred[i].item()]) for i in range(len(hh_pred))], dtype=torch.long)
    # household_sizes = household_sizes.to(device)
    # print("Done assigning household sizes using random method")
else:
    print("Done assigning household size categories using tensor extraction method")
    print(f"Using extracted size categories for {household_size_categories.size(0)} households")
    
    # Convert size categories to actual sizes for compatibility with existing functions
    # Map: 0->1, 1->2, 2->3, 3->4 (4+ category becomes 4)
    household_sizes = household_size_categories + 1  # Convert 0-3 indices to 1-4 sizes
    print("Converted size categories to actual sizes for compatibility")
    
    # Print detailed information about the conversion
    print(f"Size category distribution:")
    unique_cats, cat_counts = torch.unique(household_size_categories, return_counts=True)
    size_categories = ['1', '2', '3', '4+']
    for cat_idx, count in zip(unique_cats.cpu().numpy(), cat_counts.cpu().numpy()):
        category_name = size_categories[cat_idx] if cat_idx < len(size_categories) else f"Unknown({cat_idx})"
        print(f"  Category {cat_idx} ({category_name}): {count} households")
    
    print(f"Converted size distribution:")
    unique_sizes, size_counts = torch.unique(household_sizes, return_counts=True)
    for size, count in zip(unique_sizes.cpu().numpy(), size_counts.cpu().numpy()):
        print(f"  Size {size}: {count} households")

# Step 2: Define the GNN model
class HouseholdAssignmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_households, dropout_rate=0.05):
        super(HouseholdAssignmentGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        # Add batch normalization for better training stability
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Add dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # Enhanced final layer with residual connection
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, num_households)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # First GCN layer
        x1 = self.conv1(x, edge_index)
        x1 = self.batch_norm1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        # Second GCN layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.batch_norm2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        # Third GCN layer with residual connection
        x3 = self.conv3(x2, edge_index)
        x3 = self.batch_norm3(x3)
        x3 = self.relu(x3 + x1)  # Residual connection
        x3 = self.dropout(x3)
        
        # Enhanced final layers
        out = self.fc1(x3)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
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

# Define the columns for religion, ethnicity, and household composition
# Based on actual tensor structures from generation scripts:
# person_nodes: [age, sex, religion, ethnicity, marital, qualification, household_composition] (7 columns)
# household_nodes: [household_composition, ethnicity, religion, tenure, size, rooms] (6 columns)
religion_col_persons, religion_col_households = 2, 2
ethnicity_col_persons, ethnicity_col_households = 3, 1
household_composition_col_persons, household_composition_col_households = 6, 0

# Create the graph with more flexible edge construction (match on religion, ethnicity, or household composition)
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
        # Extract religion, ethnicity, and household composition columns for efficient comparison
        person_religion = person_nodes[:, religion_col_persons]
        person_ethnicity = person_nodes[:, ethnicity_col_persons]
        person_household_composition = person_nodes[:, household_composition_col_persons]
        
        # Create matrices for pairwise comparison using broadcasting
        # Shape: (num_persons, num_persons) - True where persons have same religion/ethnicity/household_composition
        religion_match = person_religion.unsqueeze(1) == person_religion.unsqueeze(0)
        ethnicity_match = person_ethnicity.unsqueeze(1) == person_ethnicity.unsqueeze(0)
        household_composition_match = person_household_composition.unsqueeze(1) == person_household_composition.unsqueeze(0)
        
        # Combine matches: True if religion OR ethnicity OR household composition matches
        # Including household composition in edge creation is crucial for the GNN to learn
        # to group persons with similar household composition together, which significantly
        # improves household composition accuracy in the final assignment
        matches = religion_match | ethnicity_match | household_composition_match
        
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
        safe_delete_tensor(person_household_composition)
        safe_delete_tensor(religion_match)
        safe_delete_tensor(ethnicity_match)
        safe_delete_tensor(household_composition_match)
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
def compute_loss(assignments, household_sizes, person_nodes, household_nodes, religion_loss_weight=1.0, ethnicity_loss_weight=1.0, size_loss_weight=1.0, household_composition_loss_weight=1.0):
    household_counts = assignments.sum(dim=0)  # Expected people per household (soft)

    # Size loss with 4 categories (1,2,3,4) where 4 represents 4+ but is treated as 4
    sizes_float = household_sizes.float()        # targets: 1..4
    pred_counts_capped = torch.clamp(household_counts, max=4.0)
    size_loss = F.mse_loss(pred_counts_capped, sizes_float) * size_loss_weight

    # Categorical attribute alignment via NLL over assignment mass for matching classes
    eps = 1e-8

    # Religion
    religion_col_persons, religion_col_households = 2, 2
    y_religion = person_nodes[:, religion_col_persons].long()
    num_rel_classes = int(torch.max(torch.stack([
        y_religion.max(), household_nodes[:, religion_col_households].long().max()
    ])).item()) + 1
    H_rel_onehot = F.one_hot(household_nodes[:, religion_col_households].long(), num_classes=num_rel_classes).float()
    P_rel = assignments @ H_rel_onehot  # [num_persons, C]
    rel_match_prob = P_rel[torch.arange(P_rel.size(0), device=P_rel.device), y_religion]
    religion_loss = (-torch.log(rel_match_prob + eps)).mean() * religion_loss_weight

    # Ethnicity
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    y_eth = person_nodes[:, ethnicity_col_persons].long()
    num_eth_classes = int(torch.max(torch.stack([
        y_eth.max(), household_nodes[:, ethnicity_col_households].long().max()
    ])).item()) + 1
    H_eth_onehot = F.one_hot(household_nodes[:, ethnicity_col_households].long(), num_classes=num_eth_classes).float()
    P_eth = assignments @ H_eth_onehot
    eth_match_prob = P_eth[torch.arange(P_eth.size(0), device=P_eth.device), y_eth]
    ethnicity_loss = (-torch.log(eth_match_prob + eps)).mean() * ethnicity_loss_weight

    # Household composition
    household_composition_col_persons, household_composition_col_households = 6, 0
    y_hh = person_nodes[:, household_composition_col_persons].long()
    num_hh_classes = int(torch.max(torch.stack([
        y_hh.max(), household_nodes[:, household_composition_col_households].long().max()
    ])).item()) + 1
    H_hh_onehot = F.one_hot(household_nodes[:, household_composition_col_households].long(), num_classes=num_hh_classes).float()
    P_hh = assignments @ H_hh_onehot
    hh_match_prob = P_hh[torch.arange(P_hh.size(0), device=P_hh.device), y_hh]
    household_composition_loss = (-torch.log(hh_match_prob + eps)).mean() * household_composition_loss_weight

    total_loss = size_loss + religion_loss + ethnicity_loss + household_composition_loss
    return total_loss, size_loss, religion_loss, ethnicity_loss, household_composition_loss

# Step 4: Hyperparameter tuning setup
# num_epochs = 20  # Increased epochs for better convergence
num_epochs = 400  # Increased epochs for better convergence
learning_rates = [0.005]  # Test multiple learning rates
# hidden_dims = [256]  # Test multiple hidden dimensions
hidden_dims = [64]  # Test multiple hidden dimensions
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
    'ethnicity_accuracies': None,
    'stopping_epoch': None
}

# Plotting functions
def plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir):
    """Plot assignment errors similar to assignment_model2.py"""
    
    # Calculate size errors (using category-based comparison)
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in final_assignments:
        predicted_counts[household_idx] += 1
    
    # Convert to categories for error calculation
    actual_categories = torch.clamp(household_sizes - 1, 0, 3)  # Convert 1-4 sizes to 0-3 categories
    predicted_categories = torch.clamp(predicted_counts - 1, 0, 3)  # Convert 1-4+ sizes to 0-3 categories
    
    size_errors = torch.abs(predicted_categories - actual_categories).sum().item()
    
    # Calculate religion, ethnicity, and household composition errors
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    household_composition_col_persons, household_composition_col_households = 6, 0
    
    religion_errors = 0
    ethnicity_errors = 0
    household_composition_errors = 0
    
    for person_idx, household_idx in enumerate(final_assignments):
        household_idx = household_idx.item()
        
        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]
        person_household_composition = person_nodes[person_idx, household_composition_col_persons]
        
        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]
        household_composition = household_nodes[household_idx, household_composition_col_households]
        
        if person_religion != household_religion:
            religion_errors += 1
        if person_ethnicity != household_ethnicity:
            ethnicity_errors += 1
        if person_household_composition != household_composition:
            household_composition_errors += 1
    
    # Create bar graph
    plt.figure(figsize=(15, 6))
    
    categories = ['Size Errors', 'Religion Errors', 'Ethnicity Errors', 'Household Composition Errors']
    error_counts = [size_errors, religion_errors, ethnicity_errors, household_composition_errors]
    colors = ['lightcoral', 'skyblue', 'lightgreen', 'gold']
    
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

def plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, household_composition_accuracies, output_dir):
    """Plot accuracy over epochs with religion, ethnicity, and household composition graphs"""
    
    # Make plot vertically shorter
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # Plot (a) Religion
    bars1 = ax1.bar(epoch_numbers[::10], religion_accuracies[::10], color='steelblue', alpha=0.7, width=8)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Percentage of correctly assigned persons (%)')
    ax1.set_title('(a) Religion')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels rotated 90° just above each bar (every 10th epoch)
    for i, (epoch, acc) in enumerate(zip(epoch_numbers[::10], religion_accuracies[::10])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax1.text(epoch, min(acc + 1.5, 98), f'{acc:.1f}', ha='center', va='bottom', rotation=90, fontsize=9, fontweight='bold')
    
    # Plot (b) Ethnicity
    bars2 = ax2.bar(epoch_numbers[::10], ethnicity_accuracies[::10], color='steelblue', alpha=0.7, width=8)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Percentage of correctly assigned persons (%)')
    ax2.set_title('(b) Ethnicity')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels rotated 90° just above each bar (every 10th epoch)
    for i, (epoch, acc) in enumerate(zip(epoch_numbers[::10], ethnicity_accuracies[::10])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax2.text(epoch, min(acc + 1.5, 98), f'{acc:.1f}', ha='center', va='bottom', rotation=90, fontsize=9, fontweight='bold')
    
    # Plot (c) Household Composition
    bars3 = ax3.bar(epoch_numbers[::10], household_composition_accuracies[::10], color='gold', alpha=0.7, width=8)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Percentage of correctly assigned persons (%)')
    ax3.set_title('(c) Household Composition')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels rotated 90° just above each bar (every 10th epoch)
    for i, (epoch, acc) in enumerate(zip(epoch_numbers[::10], household_composition_accuracies[::10])):
        if i % 2 == 0:  # Show every other label to avoid crowding
            ax3.text(epoch, min(acc + 1.5, 98), f'{acc:.1f}', ha='center', va='bottom', rotation=90, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_over_epochs.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Accuracy over epochs plot saved to: {accuracy_plot_path}")

# Household Size Accuracy Function (updated for 4-category system)
def calculate_size_distribution_accuracy(assignments, household_sizes):
    """
    Calculate household size distribution accuracy by comparing predicted vs expected size distributions.
    Updated to work with the new 4-category system (1, 2, 3, 4+).
    """
    # Step 1: Calculate the predicted sizes (how many people in each household)
    predicted_counts = torch.zeros_like(household_sizes, device=household_sizes.device)
    for household_idx in assignments:
        predicted_counts[household_idx] += 1  # Increment for each assignment
    
    # Step 2: Convert actual sizes to categories for comparison
    # Map: 1->0, 2->1, 3->2, 4->3 (4+ category is 4, map to 3)
    actual_categories = torch.clamp(household_sizes - 1, 0, 3)  # Convert 1-4 sizes to 0-3 categories
    
    # Step 3: Convert predicted counts to categories
    # Map: 1->0, 2->1, 3->2, 4+->3 (anything 4 or more maps to category 3)
    predicted_categories = torch.clamp(predicted_counts - 1, 0, 3)  # Convert 1-4+ sizes to 0-3 categories
    
    # Step 4: Calculate bincount of the categories (4 categories: 0, 1, 2, 3)
    num_categories = 4
    predicted_distribution = torch.bincount(predicted_categories, minlength=num_categories).float()
    actual_distribution = torch.bincount(actual_categories, minlength=num_categories).float()

    # Step 5: Calculate accuracy for each category
    accuracies = torch.min(predicted_distribution, actual_distribution) / (actual_distribution + 1e-6)  # Avoid division by 0
    overall_accuracy = accuracies.mean().item()  # Average accuracy across all categories
    
    # Debug information (optional - can be removed later)
    if torch.rand(1).item() < 0.01:  # Only print 1% of the time to avoid spam
        print(f"Size accuracy debug - Predicted: {predicted_distribution.cpu().numpy()}, Actual: {actual_distribution.cpu().numpy()}")
        print(f"Category accuracies: {accuracies.cpu().numpy()}, Overall: {overall_accuracy:.4f}")

    return overall_accuracy

# Compliance Accuracy Function
def calculate_individual_compliance_accuracy(assignments, person_nodes, household_nodes):
    religion_col_persons, religion_col_households = 2, 2
    ethnicity_col_persons, ethnicity_col_households = 3, 1
    household_composition_col_persons, household_composition_col_households = 6, 0

    total_people = assignments.size(0)
    
    correct_religion_assignments = 0
    correct_ethnicity_assignments = 0
    correct_household_composition_assignments = 0

    # Loop over each person and their assigned household
    for person_idx, household_idx in enumerate(assignments):
        household_idx = household_idx.item()  # Get the household assignment for the person

        person_religion = person_nodes[person_idx, religion_col_persons]
        person_ethnicity = person_nodes[person_idx, ethnicity_col_persons]
        person_household_composition = person_nodes[person_idx, household_composition_col_persons]

        household_religion = household_nodes[household_idx, religion_col_households]
        household_ethnicity = household_nodes[household_idx, ethnicity_col_households]
        household_composition = household_nodes[household_idx, household_composition_col_households]

        # Check if the person's religion matches the household's religion
        if person_religion == household_religion:
            correct_religion_assignments += 1

        # Check if the person's ethnicity matches the household's ethnicity
        if person_ethnicity == household_ethnicity:
            correct_ethnicity_assignments += 1

        # Check if the person's household composition matches the household's composition
        if person_household_composition == household_composition:
            correct_household_composition_assignments += 1

    religion_compliance = correct_religion_assignments / total_people
    ethnicity_compliance = correct_ethnicity_assignments / total_people
    household_composition_compliance = correct_household_composition_assignments / total_people

    return religion_compliance, ethnicity_compliance, household_composition_compliance

# Combined Accuracy Function
def calculate_all_accuracies(assignments, person_nodes, household_nodes, household_sizes):
    """
    Calculate all accuracies: religion, ethnicity, household composition, and household size.
    Returns individual accuracies and overall average accuracy.
    Updated for 4-category household size system.
    """
    # Calculate religion, ethnicity, and household composition accuracies
    religion_compliance, ethnicity_compliance, household_composition_compliance = calculate_individual_compliance_accuracy(
        assignments, person_nodes, household_nodes
    )
    
    # Calculate household size distribution accuracy (updated for 4-category system)
    size_distribution_accuracy = calculate_size_distribution_accuracy(
        assignments, household_sizes
    )
    
    # Calculate overall average accuracy (now including household composition)
    overall_accuracy = (religion_compliance + ethnicity_compliance + household_composition_compliance + size_distribution_accuracy) / 4.0
    
    return {
        'religion_compliance': religion_compliance,
        'ethnicity_compliance': ethnicity_compliance,
        'household_composition_compliance': household_composition_compliance,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    
    # Use a simpler scheduler to avoid compatibility issues
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        use_scheduler = True
    except Exception as e:
        print(f"Warning: Could not create ReduceLROnPlateau scheduler: {e}")
        print("Continuing without learning rate scheduling...")
        scheduler = None
        use_scheduler = False
    
    tau = 1.0

    monitor_memory_usage(device, "after model creation")

    # Track accuracies over epochs and convergence data
    religion_accuracies = []
    ethnicity_accuracies = []
    household_composition_accuracies = []
    epoch_numbers = []
    
    # Track convergence data for all training runs
    convergence_data = {
        'epochs': [],
        'losses': [],
        'size_losses': [],
        'religion_losses': [],
        'ethnicity_losses': [],
        'household_composition_losses': [],
        'religion_accuracies': [],
        'ethnicity_accuracies': [],
        'household_composition_accuracies': [],
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
    
    # Early stopping
    patience = 100
    patience_counter = 0
    
    # Track stopping epoch
    stopping_epoch = num_epochs  # Default to max epochs if no early stopping

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        logits = model(person_nodes, edge_index)
        assignments = gumbel_softmax(logits, tau=tau, hard=False)

        total_loss, size_loss, religion_loss, ethnicity_loss, household_composition_loss = compute_loss(
            assignments,
            household_sizes,
            person_nodes,
            household_nodes,
            religion_loss_weight=1.0,
            ethnicity_loss_weight=1.0,
            size_loss_weight=2.0,
            household_composition_loss_weight=2.0
        )
        total_loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate scheduler if available
        if use_scheduler and scheduler is not None:
            scheduler.step(total_loss)
        
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
        convergence_data['household_composition_losses'].append(household_composition_loss.item())
        convergence_data['religion_accuracies'].append(accuracies['religion_compliance'] * 100)
        convergence_data['ethnicity_accuracies'].append(accuracies['ethnicity_compliance'] * 100)
        convergence_data['household_composition_accuracies'].append(accuracies['household_composition_compliance'] * 100)
        convergence_data['size_distribution_accuracies'].append(accuracies['size_distribution_accuracy'] * 100)
        convergence_data['overall_accuracies'].append(accuracies['overall_accuracy'] * 100)
        convergence_data['epoch_time_seconds'].append(epoch_duration)
        convergence_data['cumulative_time_seconds'].append(cumulative_time)
        convergence_data['tau_values'].append(tau)
        
        # Track for detailed results
        epoch_numbers.append(epoch + 1)
        religion_accuracies.append(accuracies['religion_compliance'] * 100)
        ethnicity_accuracies.append(accuracies['ethnicity_compliance'] * 100)
        household_composition_accuracies.append(accuracies['household_composition_compliance'] * 100)
        
        # Store best epoch state and early stopping
        if total_loss.item() < best_epoch_loss:
            best_epoch_loss = total_loss.item()
            best_epoch_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= patience:
            stopping_epoch = epoch + 1
            print(f"\n    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # Clear intermediate tensors to free memory
        safe_delete_tensor(logits)
        safe_delete_tensor(assignments)
        safe_delete_tensor(final_assignments)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Calculate loss ratios for debugging
            total_loss_val = total_loss.item()
            size_ratio = size_loss.item() / total_loss_val if total_loss_val > 0 else 0
            religion_ratio = religion_loss.item() / total_loss_val if total_loss_val > 0 else 0
            ethnicity_ratio = ethnicity_loss.item() / total_loss_val if total_loss_val > 0 else 0
            household_composition_ratio = household_composition_loss.item() / total_loss_val if total_loss_val > 0 else 0
            
            print(f"\r    Epoch {epoch+1:3d}/{num_epochs} | Total: {total_loss_val:.6f} | Size: {size_loss.item():.6f}({size_ratio:.1%}) | Religion: {religion_loss.item():.6f}({religion_ratio:.1%}) | Ethnicity: {ethnicity_loss.item():.6f}({ethnicity_ratio:.1%}) | HH_Comp: {household_composition_loss.item():.6f}({household_composition_ratio:.1%}) | Religion Acc: {accuracies['religion_compliance']*100:.2f}% | Ethnicity Acc: {accuracies['ethnicity_compliance']*100:.2f}% | HH_Comp Acc: {accuracies['household_composition_compliance']*100:.2f}% | Size Acc: {accuracies['size_distribution_accuracy']*100:.2f}% | Overall Acc: {accuracies['overall_accuracy']*100:.2f}% | Tau: {tau:.3f}", end="", flush=True)
            
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
            'ethnicity_accuracies': ethnicity_accuracies.copy(),
            'household_composition_accuracies': household_composition_accuracies.copy(),
            'stopping_epoch': stopping_epoch
        })

    # Clear model and intermediate tensors from GPU memory before returning
    safe_delete_tensor(logits)
    safe_delete_tensor(assignments)
    safe_delete_model(model)
    clear_gpu_memory()
    
    monitor_memory_usage(device, "after model cleanup")
    
    if return_detailed_results:
        return best_epoch_loss, final_assignments, epoch_numbers, religion_accuracies, ethnicity_accuracies, convergence_data, final_accuracies, stopping_epoch
    else:
        return best_epoch_loss, convergence_data, final_accuracies, stopping_epoch

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
                
                final_loss, convergence_data, final_accuracies, stopping_epoch = train_model(learning_rate=lr, hidden_channels=hidden_dim)
                
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
                    'training_time': combination_training_time_str,
                    'stopping_epoch': stopping_epoch
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
                    'num_epochs': num_epochs,
                    'stopping_epoch': stopping_epoch
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
                        'overall_accuracy': final_accuracies['overall_accuracy'],
                        'stopping_epoch': stopping_epoch
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
if best_params:
    print(f"Best hyperparameters: {best_params} with final loss {best_loss}")
else:
    print("No successful training runs completed. No best hyperparameters available.")
    best_params = {
        'learning_rate': None,
        'hidden_channels': None,
        'religion_compliance': 0.0,
        'ethnicity_compliance': 0.0,
        'size_distribution_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'stopping_epoch': 0
    }
    best_loss = float('inf')

# Save hyperparameter tuning results
print(f"\nSaving hyperparameter tuning results to: {output_dir}")

# Save basic results as CSV
if hp_results:
    hp_results_df = pd.DataFrame(hp_results)
    # Mark the best hyperparameter combination
    if best_model_info.get('lr') is not None and best_model_info.get('hidden_channels') is not None:
        hp_results_df['is_best'] = (
            (hp_results_df['learning_rate'] == best_model_info['lr']) &
            (hp_results_df['hidden_channels'] == best_model_info['hidden_channels'])
        )
    else:
        hp_results_df['is_best'] = False
    hp_results_path = os.path.join(output_dir, 'hp_tuning_results.csv')
    hp_results_df.to_csv(hp_results_path, index=False)
    print(f"Basic results saved: {hp_results_path}")
else:
    print("No results to save - no successful training runs")

# Save detailed results as CSV
if detailed_results:
    detailed_results_df = pd.DataFrame(detailed_results)
    detailed_results_path = os.path.join(output_dir, 'detailed_hp_results.csv')
    detailed_results_df.to_csv(detailed_results_path, index=False)
    print(f"Detailed results saved: {detailed_results_path}")
else:
    print("No detailed results to save - no successful training runs")

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
else:
    print("No convergence data to save - no successful training runs")

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
    'best_learning_rate': best_params.get('learning_rate'),
    'best_hidden_channels': best_params.get('hidden_channels'),
    'best_loss': best_loss,
    'best_religion_compliance': best_params.get('religion_compliance', 0.0),
    'best_ethnicity_compliance': best_params.get('ethnicity_compliance', 0.0),
    'best_size_distribution_accuracy': best_params.get('size_distribution_accuracy', 0.0),
    'best_overall_accuracy': best_params.get('overall_accuracy', 0.0),
    'best_stopping_epoch': best_params.get('stopping_epoch', 0)
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
best_params_with_loss['religion_accuracy_percent'] = best_params.get('religion_compliance', 0.0) * 100
best_params_with_loss['ethnicity_accuracy_percent'] = best_params.get('ethnicity_compliance', 0.0) * 100
best_params_with_loss['size_distribution_accuracy_percent'] = best_params.get('size_distribution_accuracy', 0.0) * 100
best_params_with_loss['overall_accuracy_percent'] = best_params.get('overall_accuracy', 0.0) * 100
best_params_with_loss['total_training_time'] = total_training_time_str
best_params_with_loss['stopping_epoch'] = best_params.get('stopping_epoch', 0)

best_params_path = os.path.join(output_dir, 'best_hyperparameters.json')
with open(best_params_path, 'w') as f:
    json.dump(best_params_with_loss, f, indent=4)
print(f"Best parameters saved: {best_params_path}")

print(f"\nAll hyperparameter tuning results saved to: {output_dir}")

# Use saved best model information instead of retraining
print(f"\n{'='*60}")
print(f"USING BEST MODEL RESULTS FOR PLOTTING (NO RETRAINING)")
print(f"{'='*60}")

# Check if we have valid best model information
if best_model_info['model_state'] is not None:
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
    household_composition_accuracies = best_model_info['household_composition_accuracies']
    best_convergence_data = best_model_info['convergence_data']
    final_accuracies = best_model_info['detailed_accuracies']

    print(f"\nUsing saved best model results (no retraining needed)")

    # Generate plots using saved data
    print("\nGenerating plots...")
    plot_assignment_errors(final_assignments, household_sizes, person_nodes, household_nodes, output_dir)
    plot_accuracy_over_epochs(epoch_numbers, religion_accuracies, ethnicity_accuracies, household_composition_accuracies, output_dir)
else:
    print("\nNo successful training runs completed. Cannot generate plots.")
    print("Please check the error messages above and fix the issues before running again.")
    exit(1)

# Save final assignment results
if best_model_info['model_state'] is not None:
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
    best_params_with_loss['final_household_composition_compliance'] = final_accuracies['household_composition_compliance']
    best_params_with_loss['final_size_distribution_accuracy'] = final_accuracies['size_distribution_accuracy']
    best_params_with_loss['final_overall_accuracy'] = final_accuracies['overall_accuracy']
    best_params_with_loss['final_religion_accuracy_percent'] = final_accuracies['religion_compliance'] * 100
    best_params_with_loss['final_ethnicity_accuracy_percent'] = final_accuracies['ethnicity_compliance'] * 100
    best_params_with_loss['final_household_composition_accuracy_percent'] = final_accuracies['household_composition_compliance'] * 100
    best_params_with_loss['final_size_distribution_accuracy_percent'] = final_accuracies['size_distribution_accuracy'] * 100
    best_params_with_loss['final_overall_accuracy_percent'] = final_accuracies['overall_accuracy'] * 100

    # Re-save updated best parameters
    with open(best_params_path, 'w') as f:
        json.dump(best_params_with_loss, f, indent=4)

    print(f"\nFinal Results with Best Hyperparameters:")
    print(f"  Learning Rate: {best_model_info['lr']}")
    print(f"  Hidden Channels: {best_model_info['hidden_channels']}")
    print(f"  Final Loss: {best_model_info['loss']:.6f}")
    print(f"  Stopping Epoch: {best_model_info.get('stopping_epoch', 'N/A')}")
    print(f"  Religion Compliance: {final_accuracies['religion_compliance'] * 100:.2f}%")
    print(f"  Ethnicity Compliance: {final_accuracies['ethnicity_compliance'] * 100:.2f}%")
    print(f"  Household Composition Compliance: {final_accuracies['household_composition_compliance'] * 100:.2f}%")
    print(f"  Size Distribution Accuracy: {final_accuracies['size_distribution_accuracy'] * 100:.2f}%")
    print(f"  Overall Accuracy: {final_accuracies['overall_accuracy'] * 100:.2f}%")
    print(f"  Results and plots saved to: {output_dir}")
else:
    print(f"\nNo final results to save - no successful training runs")

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