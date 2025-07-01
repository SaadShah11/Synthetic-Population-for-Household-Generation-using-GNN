import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphNorm
import random
import json
import time
from datetime import timedelta
import numpy as np
from collections import Counter
import shutil  # For directory operations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import argparse
import geopandas as gpd  # Added for geo plotting
import plotly.express as px  # Added for geo plotting

# Add argument parser for command line parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate synthetic households using GNN')
    parser.add_argument('--area_code', type=str, required=True,
                       help='Oxford area code to process (e.g., E02005924)')
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
selected_area_code = args.area_code

print(f"Running Household Generation for area: {selected_area_code}")

# Set display options permanently
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Device selection with better fallback options
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=torch.inf)

def create_geo_plot_trace(selected_area_code, current_dir):
    """
    Create a geo plot trace showing all areas in white except the selected area code which is shaded.
    Returns the geo traces that can be added to a subplot.
    
    Parameters:
    selected_area_code - The area code to highlight
    current_dir - The current directory path for file loading
    """
    try:
        # Define paths relative to the code directory
        BASE = os.path.join(current_dir, '../data/')
        PERSONS_DIR = os.path.join(current_dir, '../data/preprocessed-data/individuals')
        
        # Load age data and rename the geography code
        age_file = os.path.join(PERSONS_DIR, "Age_Perfect_5yrs.csv")
        if not os.path.exists(age_file):
            print(f"Warning: Age data file not found at {age_file}")
            return [], {}
            
        age = pd.read_csv(age_file)
        age = age.rename(columns={"geography code": "MSOA21CD"})
        
        # Load shapefiles
        msoa_fp = os.path.join(BASE, "geodata", "MSOA_2021_EW_BGC_V3.shp")
        red_fp = os.path.join(BASE, "geodata", "boundary.geojson")
        
        if not os.path.exists(msoa_fp) or not os.path.exists(red_fp):
            print(f"Warning: Geodata files not found. Expected at {msoa_fp} and {red_fp}")
            return [], {}
        
        # Read in spatial data
        gdf_msoa = gpd.read_file(msoa_fp).to_crs(4326)
        red_bnd = gpd.read_file(red_fp).to_crs(4326)
        
        # Merge population totals
        gdf_msoa = gdf_msoa.merge(age[["MSOA21CD", "total"]], on="MSOA21CD", how="left")
        gdf_msoa["total"] = gdf_msoa["total"].fillna(0)
        
        # Manually remove unwanted MSOAs
        exclude_codes = [
            "E02005939", "E02005979", "E02005963", "E02005959"
        ]
        gdf_msoa = gdf_msoa[~gdf_msoa["MSOA21CD"].isin(exclude_codes)]
        
        # Clip to red boundary
        red_union = red_bnd.unary_union
        gdf_clip = gdf_msoa[gdf_msoa.intersects(red_union)].copy()
        
        # Create color column: selected area gets color 1, others get color 0
        gdf_clip["color_value"] = gdf_clip["MSOA21CD"].apply(
            lambda x: 1 if x == selected_area_code else 0
        )
        
        # Compute accurate centroids for labels
        proj_crs = 27700
        gdf_proj = gdf_clip.to_crs(proj_crs)
        centroids = gdf_proj.geometry.centroid.to_crs(4326)
        gdf_clip["lon"] = centroids.x
        gdf_clip["lat"] = centroids.y
        
        # Create traces list
        traces = []
        
        # Add choropleth trace
        choropleth_trace = go.Choropleth(
            geojson=json.loads(gdf_clip.to_json()),
            locations=gdf_clip["MSOA21CD"],
            featureidkey="properties.MSOA21CD",
            z=gdf_clip["color_value"],
            colorscale=[[0, "white"], [1, "lightblue"]],  # White for 0, light blue for selected
            showscale=False,
            hovertemplate="<b>%{location}</b><extra></extra>",
            name="Areas"
        )
        traces.append(choropleth_trace)
        
        # Add red boundary outline
        for poly in red_bnd.geometry.explode(index_parts=False):
            boundary_trace = go.Scattergeo(
                lon=list(poly.exterior.coords.xy[0]),
                lat=list(poly.exterior.coords.xy[1]),
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo="skip"
            )
            traces.append(boundary_trace)
        
        # Calculate bounds for the geo layout with reduced padding for bigger geo plots
        # Use red boundary bounds instead of just clipped areas for better coverage
        red_bounds = red_bnd.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = red_bounds
        
        # Reduce padding to make geo plots bigger within their allocated space
        lat_padding = (lat_max - lat_min) * 0.05  # Further reduced to 5% padding for larger geo plot
        lon_padding = (lon_max - lon_min) * 0.05  # Further reduced to 5% padding for larger geo plot
        
        geo_layout = {
            'visible': False,
            'lataxis_range': [lat_min - lat_padding, lat_max + lat_padding],
            'lonaxis_range': [lon_min - lon_padding, lon_max + lon_padding],
            'projection_type': 'mercator'
        }
        
        return traces, geo_layout
        
    except Exception as e:
        print(f"Warning: Could not create geo plot trace: {e}")
        return [], {}

def get_target_tensors(cross_table, hh_categories, hh_map, feature_categories, feature_map):
    y_hh = torch.zeros(num_households, dtype=torch.long, device=device)
    y_feature = torch.zeros(num_households, dtype=torch.long, device=device)
    
    # Populate target tensors based on the cross-table and categories
    # Changed order to match new glossary: household compositions first, then ethnicity/religion
    household_idx = 0

    for _, row in cross_table.iterrows():
        for hh in hh_categories:  # household compositions
            for feature in feature_categories:  # ethnicity/religion categories
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
ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Ethnicity.csv'))
religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/Religion.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition.csv'))
# hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Households.csv'))
hhcomp_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/individuals/HH_composition_Updated.csv'))
# hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity.csv'))
# hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion.csv'))
hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion_Updated.csv'))
hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_Updated.csv'))

# Use the area code passed from command line
oxford_areas = [selected_area_code]
print(f"Processing Oxford area: {oxford_areas[0]}")

ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
religion_categories = ['C','B','H','J','M','S','O','N','NS']

# Filter the DataFrame for the specified Oxford areas
ethnicity_df = ethnicity_df[ethnicity_df['geography code'].isin(oxford_areas)]
religion_df = religion_df[religion_df['geography code'].isin(oxford_areas)]
hhcomp_df = hhcomp_df[hhcomp_df['geography code'].isin(oxford_areas)]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'].isin(oxford_areas)]
hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'].isin(oxford_areas)]

num_households = int(hhcomp_df['total'].iloc[0])
print(f"Number of households: {num_households}")

# Preprocess household composition data
# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-1C'] + hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-1C'] + hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-1C'] + hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-1C'] + hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-1C', '1FM-nC', '1FC-1C', '1FC-nC', '1FL-1C', '1FL-nC', '1H-1C', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(['1FM', '1FC', '1FL'], axis=1)

# hhcomp_df['1FM-2C'] = hhcomp_df['1FM-nC']
# hhcomp_df['1FC-2C'] = hhcomp_df['1FC-nC']
# hhcomp_df['1FL-2C'] = hhcomp_df['1FL-nC']
# hhcomp_df['1H-2C'] = hhcomp_df['1H-nC']
# hhcomp_df.drop(columns=['1FM-nC', '1FC-nC', '1FL-nC', '1H-nC', 'total', 'geography code'], inplace=True)
# hhcomp_df = hhcomp_df.drop(columns=['total', 'geography code'], inplace=True)

hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
# hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-nC', '1FM-nA','1FC-0C','1FC-nC','1FC-nA','1FL-nA','1FL-nC','1H-nS','1H-nE','1H-nA', '1H-nC']

# Filter and preprocess columns
# filtered_columns = [col for col in hhcomp_by_ethnicity_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[filtered_columns]
# filtered_columns = [col for col in hhcomp_by_religion_df.columns if not any(substring in col for substring in ['OF-Married', 'OF-Cohabiting', 'OF-LoneParent'])]
# hhcomp_by_religion_df = hhcomp_by_religion_df[filtered_columns]
hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns = ['total', 'geography code'])
hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns = ['total', 'geography code'])

# Encode the categories to indices
ethnicity_map = {category: i for i, category in enumerate(ethnicity_categories)}
religion_map = {category: i for i, category in enumerate(religion_categories)}
hh_map = {category: i for i, category in enumerate(hh_compositions)}

# Create household nodes with unique IDs
households_nodes = torch.arange(num_households).view(num_households, 1).to(device)

# Create nodes for ethnicity and religion categories
ethnicity_nodes = torch.tensor([[ethnicity_map[ethnicity]] for ethnicity in ethnicity_categories], dtype=torch.float).to(device)
religion_nodes = torch.tensor([[religion_map[religion]] for religion in religion_categories], dtype=torch.float).to(device)

# Combine all nodes into a single tensor
node_features = torch.cat([households_nodes, ethnicity_nodes, religion_nodes], dim=0).to(device)

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
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    return edge_index

# Generate edge index
edge_index = generate_edge_index(num_households)

# Create the data object for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index).to(device)

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
        self.dropout = torch.nn.Dropout(0.08)
        
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
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.graph_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.graph_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
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
# learning_rates = [0.001, 0.0005, 0.0001]
learning_rates = [0.001]
hidden_channel_options = [64, 128, 256]
mlp_hidden_dim = 256
num_epochs = 2000

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

# Optimized GPU-friendly accuracy function for multi-task learning
def calculate_distribution_task_accuracy(pred_1, pred_2, target_combination, actual_crosstable):
    """
    Fast GPU-optimized distribution-based accuracy calculation.
    Uses tensor operations instead of pandas for speed during training.
    """
    categories_1, categories_2 = target_combination
    
    # Map attribute names to category counts
    category_sizes = {
        'hhcomp': len(hh_compositions),
        'ethnicity': len(ethnicity_categories),
        'religion': len(religion_categories)
    }
    
    size_1 = category_sizes[categories_1]
    size_2 = category_sizes[categories_2]
    
    # Create predicted counts tensor (keep on GPU)
    # Use a flattened approach with new ordering: combination_idx = pred_1 * size_2 + pred_2
    combo_indices = pred_1 * size_2 + pred_2
    total_combinations = size_1 * size_2
    
    # Count occurrences efficiently on GPU
    predicted_counts = torch.bincount(combo_indices, minlength=total_combinations).float()
    
    # Pre-compute actual counts tensor (do this only once, not every epoch)
    cache_key = f'actual_counts_{categories_1}_{categories_2}'
    if not hasattr(calculate_distribution_task_accuracy, cache_key):
        # Extract actual counts and convert to tensor format
        actual_counts_tensor = torch.zeros(total_combinations, dtype=torch.float, device=device)
        
        category_map = {
            'hhcomp': hh_compositions,
            'ethnicity': ethnicity_categories,
            'religion': religion_categories
        }
        
        cats_1 = category_map[categories_1]
        cats_2 = category_map[categories_2]
        
        # Changed order to match new glossary: household compositions first, then ethnicity/religion
        for i1, cat1 in enumerate(cats_1):  # household compositions
            for i2, cat2 in enumerate(cats_2):  # ethnicity/religion categories
                original_col = f'{cat1} {cat2}'
                if original_col in actual_crosstable.columns:
                    combo_idx = i1 * size_2 + i2
                    actual_counts_tensor[combo_idx] = actual_crosstable[original_col].iloc[0]
        
        # Cache the result to avoid recomputation
        setattr(calculate_distribution_task_accuracy, cache_key, actual_counts_tensor)
    
    actual_counts = getattr(calculate_distribution_task_accuracy, cache_key)
    
    # Calculate R² efficiently on GPU
    actual_mean = actual_counts.mean()
    ss_tot = torch.sum((actual_counts - actual_mean) ** 2)
    ss_res = torch.sum((actual_counts - predicted_counts) ** 2)
    
    if ss_tot > 1e-12:
        r2 = 1.0 - (ss_res / ss_tot)
        return max(0.0, r2.item())  # Ensure non-negative and convert to Python float
    else:
        return 1.0

# Function to calculate RMSE
def calculate_rmse(generated_counts, target_counts):
    """
    Calculate RMSE between two distributions (dicts of category: count).
    """
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)
    mse = np.mean((gen_vals - tgt_vals) ** 2)
    return np.sqrt(mse)

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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom loss function
    def custom_loss_function(hh_out, feature_out, y_hh, y_feature):
        loss_hh = F.cross_entropy(hh_out, y_hh) 
        loss_feature = F.cross_entropy(feature_out, y_feature)
        total_loss = loss_hh + loss_feature
        return total_loss

    loss_data = {}
    accuracy_data = {}
    best_epoch_loss = float('inf')
    best_epoch_state = None
    
    # Storage for tracking metrics across epochs
    epoch_accuracies = []
    convergence_data = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
        'cumulative_time_seconds': [],
        'epoch_time_seconds': []
    }
    
    # Start timing for epoch-wise tracking
    training_start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
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
        
        # Calculate losses for all target combinations
        for i in range(len(targets)):
            current_loss = custom_loss_function(
                out[targets[i][0][0]], out[targets[i][0][1]],
                targets[i][1][0], targets[i][1][1]
            )
            loss += current_loss
            
        # Calculate accuracy only every 100 epochs to speed up training
        if (epoch + 1) % 100 == 0:
            epoch_task_accuracies = []
            for i in range(len(targets)):
                # Calculate distribution-based accuracy for this task
                pred_1 = out[targets[i][0][0]].argmax(dim=1)
                pred_2 = out[targets[i][0][1]].argmax(dim=1)
                
                # Get the corresponding actual cross-table
                if i == 0:  # hhcomp-religion
                    actual_crosstable = hhcomp_by_religion_df
                else:  # hhcomp-ethnicity
                    actual_crosstable = hhcomp_by_ethnicity_df
                
                task_distribution_accuracy = calculate_distribution_task_accuracy(
                    pred_1, pred_2, targets[i][0], actual_crosstable
                )
                epoch_task_accuracies.append(task_distribution_accuracy)

            # Calculate average accuracy for this epoch
            avg_epoch_accuracy = sum(epoch_task_accuracies) / len(epoch_task_accuracies)
            epoch_accuracies.append(avg_epoch_accuracy)
            
            # Print metrics every 100 epochs
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Distribution Accuracy: {avg_epoch_accuracy:.4f}')
        
        # Store best epoch state
        if loss.item() < best_epoch_loss:
            best_epoch_loss = loss.item()
            best_epoch_state = model.state_dict().copy()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time = epoch_end_time - training_start_time

        # Store loss data for each epoch
        loss_data[epoch] = loss.item()
        
        # Store convergence data
        convergence_data['epochs'].append(epoch + 1)
        convergence_data['losses'].append(loss.item())
        convergence_data['epoch_time_seconds'].append(epoch_duration)
        convergence_data['cumulative_time_seconds'].append(cumulative_time)
        
        # Store accuracy data (only when calculated)
        if (epoch + 1) % 100 == 0:
            convergence_data['accuracies'].append(avg_epoch_accuracy)
        else:
            convergence_data['accuracies'].append(None)  # Placeholder for missing accuracy

    # Calculate average accuracy across all epochs
    average_accuracy = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0

    # Load best epoch state for evaluation
    model.load_state_dict(best_epoch_state)
    
    # Evaluate predictions
    model.eval()
    with torch.no_grad():
        hh_out, ethnicity_out, religion_out = model(data)
        
        out = {}
        out['hhcomp'] = hh_out[:num_households]
        out['ethnicity'] = ethnicity_out[:num_households]
        out['religion'] = religion_out[:num_households]
        
        # Get the predicted class for each attribute by taking argmax of output logits
        hh_pred = out['hhcomp'].argmax(dim=1)
        ethnicity_pred = out['ethnicity'].argmax(dim=1)
        religion_pred = out['religion'].argmax(dim=1)
        
        # Calculate distribution-based accuracy across all tasks
        net_accuracy = 0
        final_task_accuracies = {}
        
        for i in range(len(targets)):
            # Get predictions for the current target combination (e.g., hhcomp+religion or hhcomp+ethnicity)
            pred_1 = out[targets[i][0][0]].argmax(dim=1)  # First attribute prediction (e.g., household composition)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)  # Second attribute prediction (e.g., religion or ethnicity)
            
            # Get the corresponding actual cross-table
            if i == 0:  # hhcomp-religion
                actual_crosstable = hhcomp_by_religion_df
            else:  # hhcomp-ethnicity
                actual_crosstable = hhcomp_by_ethnicity_df
            
            # Calculate distribution-based accuracy (R²)
            task_distribution_accuracy = calculate_distribution_task_accuracy(
                pred_1, pred_2, targets[i][0], actual_crosstable
            )
            
            # Accumulate accuracy across all target combinations
            net_accuracy += task_distribution_accuracy
            task_name = '_'.join(targets[i][0])
            final_task_accuracies[task_name] = task_distribution_accuracy * 100
        
        final_accuracy = net_accuracy / len(targets)
        
        # Print final task accuracies
        print(f"\n=== DISTRIBUTION-BASED ACCURACY RESULTS ===")
        for task, acc in final_task_accuracies.items():
            print(f"{task} distribution accuracy (R²): {acc:.2f}%")
        print(f"Overall distribution accuracy: {final_accuracy*100:.2f}%")
        
        # Calculate RMSE
        net_rmse = 0
        final_task_rmses = {}
        for i in range(len(targets)):
            pred_1 = out[targets[i][0][0]].argmax(dim=1)
            pred_2 = out[targets[i][0][1]].argmax(dim=1)
            if i == 0:
                actual_crosstable = hhcomp_by_religion_df
            else:
                actual_crosstable = hhcomp_by_ethnicity_df
            # Calculate R2
            task_distribution_accuracy = calculate_distribution_task_accuracy(
                pred_1, pred_2, targets[i][0], actual_crosstable
            )
            # Calculate RMSE
            # Build predicted and actual counts dicts
            size_1 = len(hh_compositions if targets[i][0][0]=='hhcomp' else (ethnicity_categories if targets[i][0][0]=='ethnicity' else religion_categories))
            size_2 = len(religion_categories if targets[i][0][1]=='religion' else ethnicity_categories)
            pred_counts = torch.bincount(pred_1 * size_2 + pred_2, minlength=size_1*size_2).cpu().numpy()
            # Build actual counts
            actual_counts = []
            cats_1 = hh_compositions if targets[i][0][0]=='hhcomp' else (ethnicity_categories if targets[i][0][0]=='ethnicity' else religion_categories)
            cats_2 = religion_categories if targets[i][0][1]=='religion' else ethnicity_categories
            for cat1 in cats_1:
                for cat2 in cats_2:
                    col = f'{cat1} {cat2}'
                    actual_counts.append(actual_crosstable[col].iloc[0] if col in actual_crosstable.columns else 0)
            pred_dict = {j: pred_counts[j] for j in range(len(pred_counts))}
            actual_dict = {j: actual_counts[j] for j in range(len(actual_counts))}
            task_rmse = calculate_rmse(pred_dict, actual_dict)
            net_rmse += task_rmse
            task_name = '_'.join(targets[i][0])
            final_task_rmses[task_name] = task_rmse
        overall_rmse = net_rmse / len(targets)
        # Print RMSE results
        print(f"\n=== RMSE RESULTS ===")
        for task, rmse in final_task_rmses.items():
            print(f"{task} RMSE: {rmse:.2f}")
        print(f"Overall RMSE: {overall_rmse:.2f}")
        
        # Update best model info if this model performs better
        global best_model_info
        if final_accuracy > best_model_info['accuracy'] or (final_accuracy == best_model_info['accuracy'] and best_epoch_loss < best_model_info['loss']):
            best_model_info.update({
                'model_state': best_epoch_state,
                'loss': best_epoch_loss,
                'accuracy': final_accuracy,
                'predictions': (hh_pred, ethnicity_pred, religion_pred),
                'lr': lr,
                'hidden_channels': hidden_channels,
                'convergence_data': convergence_data,
                'rmse': overall_rmse,
                'task_rmses': final_task_rmses
            })
    
    return best_epoch_loss, average_accuracy, final_accuracy, (hh_pred, ethnicity_pred, religion_pred), convergence_data

# Run grid search over hyperparameters
total_start_time = time.time()

for lr in learning_rates:
    for hidden_channels in hidden_channel_options:
        print(f"Training with lr={lr}, hidden_channels={hidden_channels}")
        
        # Start timing for this combination
        start_time = time.time()
        
        # Train the model for the current combination of hyperparameters
        final_loss, average_accuracy, final_accuracy, predictions, convergence_data = train_model(lr, hidden_channels, num_epochs, data, targets)
        
        # End timing for this combination
        end_time = time.time()
        train_time = end_time - start_time
        train_time_str = str(timedelta(seconds=int(train_time)))
        
        # Store the results
        results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'final_loss': final_loss,
            'average_accuracy': final_accuracy,
            'training_time': train_time_str,
            'rmse': best_model_info.get('rmse', None)
        })
        
        # Store timing results
        time_results.append({
            'learning_rate': lr,
            'hidden_channels': hidden_channels,
            'training_time': train_time_str
        })

        # Print the results for the current run
        print(f"Finished training with lr={lr}, hidden_channels={hidden_channels}")
        print(f"Final Loss: {final_loss}, Average Distribution Accuracy: {average_accuracy:.4f}, Final Distribution Accuracy: {final_accuracy:.4f}")
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

# Add best_model column to results_df using best_model_info
results_df['best_model'] = (
    (results_df['learning_rate'] == best_model_info['lr']) &
    (results_df['hidden_channels'] == best_model_info['hidden_channels'])
)

# Print best model information
print("\nBest Model Information:")
print(f"Learning Rate: {best_model_info['lr']}")
print(f"Hidden Channels: {best_model_info['hidden_channels']}")
print(f"Best Loss: {best_model_info['loss']:.4f}")
print(f"Best Distribution Accuracy (R²): {best_model_info['accuracy']:.4f}")

# Create output directory if it doesn't exist
# output_dir = os.path.join(current_dir, 'outputs')
output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
os.makedirs(output_dir, exist_ok=True)

# Save best model information and results
# output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
# os.makedirs(output_dir, exist_ok=True)

# Save best model predictions
best_predictions = {
    'household_pred': best_model_info['predictions'][0].cpu().numpy(),
    'ethnicity_pred': best_model_info['predictions'][1].cpu().numpy(),
    'religion_pred': best_model_info['predictions'][2].cpu().numpy()
}

# Save hyperparameter results
results_df.to_csv(os.path.join(output_dir, 'generateHouseholds_results.csv'), index=False)

# Save convergence data from best model
if 'convergence_data' in best_model_info:
    convergence_df = pd.DataFrame(best_model_info['convergence_data'])
    convergence_df.to_csv(os.path.join(output_dir, 'convergence_data.csv'), index=False)

# Save performance data
performance_data = {
    'area_code': selected_area_code,
    'num_households': num_households,
    'training_time_seconds': total_training_time,
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'final_accuracy': best_model_info['accuracy'],
    'rmse': best_model_info.get('rmse', None)
}
performance_df = pd.DataFrame([performance_data])
performance_df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)

# Save best model configuration
best_config = {
    'learning_rate': best_model_info['lr'],
    'hidden_channels': best_model_info['hidden_channels'],
    'loss': best_model_info['loss'],
    'accuracy': best_model_info['accuracy']
}

# Extract the best model's predictions for visualization
hh_pred, ethnicity_pred, religion_pred = best_model_info['predictions']

# Create household tensor with attributes matching original format
# Expected format: [household_composition, ethnicity, religion] (3 columns)
# Where hh_composition is at index 0, ethnicity is at index 1, religion is at index 2
household_nodes_tensor = torch.stack([
    hh_pred,        # Column 0: household composition
    ethnicity_pred, # Column 1: ethnicity
    religion_pred   # Column 2: religion
], dim=1)

# Save household tensor
household_tensor_path = os.path.join(output_dir, 'household_nodes.pt')
torch.save(household_nodes_tensor.cpu(), household_tensor_path)
print(f"\nBest model outputs saved to {output_dir}")

# Get the predicted household compositions, ethnicities, and religions
hh_comp_pred_indices = hh_pred.cpu().numpy()
ethnicity_pred_indices = ethnicity_pred.cpu().numpy()
religion_pred_indices = religion_pred.cpu().numpy()

# Convert indices to category names
hh_comp_pred_names = [hh_compositions[i] for i in hh_comp_pred_indices]
ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred_indices]
religion_pred_names = [religion_categories[i] for i in religion_pred_indices]

# Calculate counts of actual categories from the original data
hh_comp_actual = {}
for hh_comp in hh_compositions:
    hh_comp_actual[hh_comp] = hhcomp_df[hh_comp].iloc[0]

# Fix ethnicity and religion extraction based on column structure in the datasets
ethnicity_actual = {}
for eth in ethnicity_categories:
    ethnicity_actual[eth] = ethnicity_df[eth].iloc[0]

religion_actual = {}
for rel in religion_categories:
    religion_actual[rel] = religion_df[rel].iloc[0]

# Calculate counts of predicted categories
hh_comp_pred = dict(Counter(hh_comp_pred_names))
ethnicity_pred = dict(Counter(ethnicity_pred_names))
religion_pred = dict(Counter(religion_pred_names))

# Normalize the actual distributions to match the total number of households in predictions
# This ensures fair comparison of relative proportions
total_actual_ethnicity = sum(ethnicity_actual.values())
total_actual_religion = sum(religion_actual.values())
total_pred = num_households

if total_actual_ethnicity > 0:
    ethnicity_actual = {k: v * total_pred / total_actual_ethnicity for k, v in ethnicity_actual.items()}
if total_actual_religion > 0:
    religion_actual = {k: v * total_pred / total_actual_religion for k, v in religion_actual.items()}

# Now create crosstable dataframes for visualization
# Reshape actual crosstables to match our format
hh_by_ethnicity_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_actual_reshaped = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Extract the actual counts from the crosstable dataframes
for hh in hh_compositions:
    for eth in ethnicity_categories:
        col_name = f'{hh} {eth}'
        if col_name in hhcomp_by_ethnicity_df.columns:
            hh_by_ethnicity_actual_reshaped.loc[hh, eth] = hhcomp_by_ethnicity_df[col_name].iloc[0]
    
    for rel in religion_categories:
        col_name = f'{hh} {rel}'
        if col_name in hhcomp_by_religion_df.columns:
            hh_by_religion_actual_reshaped.loc[hh, rel] = hhcomp_by_religion_df[col_name].iloc[0]

# Create predicted crosstables from our predictions
hh_by_ethnicity_pred = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
hh_by_religion_pred = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)

# Fill the predicted crosstables based on our model predictions
for i in range(len(hh_comp_pred_names)):
    hh = hh_comp_pred_names[i]
    eth = ethnicity_pred_names[i]
    rel = religion_pred_names[i]
    
    hh_by_ethnicity_pred.loc[hh, eth] += 1
    hh_by_religion_pred.loc[hh, rel] += 1

# Plotly version of individual attribute distribution plots
def plotly_attribute_distributions(attribute_dicts, categories_dict, use_log=False, filter_zero_bars=False, max_cols=2, save_path=None):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for multiple attributes.
    Now includes a geo plot in the top right corner showing the selected area.
    
    Parameters:
    attribute_dicts - Dictionary of attribute names to (actual, predicted) count dictionaries
    categories_dict - Dictionary of attribute names to lists of categories
    use_log - Whether to use log scale for y-axis
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    max_cols - Maximum number of columns in the subplot grid
    save_path - Optional path to save the plot as HTML
    """
    attrs = list(attribute_dicts.keys())
    num_plots = len(attrs)
    
    # Add one extra column for the geo plot
    num_cols = min(num_plots, max_cols) + 1
    num_rows = math.ceil(num_plots / (num_cols - 1))  # Exclude geo column from calculation
    
    # Pre-calculate accuracy for each attribute
    accuracy_data = {}
    for attr_name in attrs:
        actual_dict, predicted_dict = attribute_dicts[attr_name]
        categories = categories_dict[attr_name]
        
        # Filter zero bars if requested
        if filter_zero_bars:
            filtered_cats = [
                cat for cat in categories
                if not (actual_dict.get(cat, 0) == 0 and predicted_dict.get(cat, 0) == 0)
            ]
            categories = filtered_cats
        
        # Calculate R² accuracy
        r2 = calculate_r2_accuracy(
            {cat: predicted_dict.get(cat, 0) for cat in categories},
            {cat: actual_dict.get(cat, 0) for cat in categories}
        )
        accuracy_data[attr_name] = r2 * 100.0
    
    # Create subplot specifications with geo plot spanning multiple rows
    specs = []
    for row in range(num_rows):
        row_specs = []
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Top right corner for geo plot
                row_specs.append({"type": "geo", "rowspan": min(num_rows, 3)})  # Span up to 3 rows
            elif row > 0 and row < min(num_rows, 3) and col == num_cols - 1:  # Skip cells for geo plot span
                row_specs.append(None)
            else:
                row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Create complete subplot titles with accuracy information, accounting for rowspan
    subplot_titles = []
    attr_idx = 0
    
    for row in range(num_rows):
        for col in range(num_cols):
            if row == 0 and col == num_cols - 1:  # Geo plot position (first row)
                subplot_titles.append("")  # No title for geo plot
            elif row > 0 and row < min(num_rows, 3) and col == num_cols - 1:  # Geo plot spanned rows
                subplot_titles.append(None)  # None for spanned cells
            elif attr_idx < len(attrs):  # Main plot positions
                attr_name = attrs[attr_idx]
                accuracy = accuracy_data[attr_name]
                subplot_titles.append(f"{attr_name} - Accuracy:{accuracy:.2f}%")
                attr_idx += 1
            else:  # Empty positions
                subplot_titles.append("")
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.10,
        vertical_spacing=0.20
    )
    
    # Add attribute distribution plots
    for idx, attr_name in enumerate(attrs):
        row = (idx // (num_cols - 1)) + 1  # Exclude geo column from calculation
        col = (idx % (num_cols - 1)) + 1   # Exclude geo column from calculation
        
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
    
    # Add geo plot in top right corner
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=num_cols)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=num_cols
        )
    
    # Update layout with increased height for larger geo plot
    fig.update_layout(
        height=350 * num_rows,  # Increased height to accommodate larger geo plot
        width=450 * num_cols,  # Fixed width
        title_text="Individual Attributes: Actual vs. Predicted",
        showlegend=True,
        plot_bgcolor="white",
        barmode='group',
        margin=dict(l=40, r=40, t=80, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column
            bgcolor='rgba(255,255,255,0.9)'
        )
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
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Household attributes plot saved to: {save_path}")
    
    # Display the plot
    # fig.show()

# Plotly version of crosstable plots
def plotly_crosstable_comparison(
    actual_dfs, 
    predicted_dfs, 
    titles, 
    show_keys=False, 
    num_cols=1, 
    filter_zero_bars=True,
    save_path=None
):
    """
    Creates Plotly subplots comparing actual vs. predicted distributions for crosstables.
    # Now includes a geo plot positioned above the first row crosstable. - COMMENTED OUT
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    show_keys - Whether to show full category key combinations (True) or numeric indices (False)
    num_cols - Number of columns in the subplot grid
    filter_zero_bars - Whether to filter out bars where both actual and predicted are zero
    save_path - Optional path to save the plot as HTML
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    rmse_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays for bar charts
        actual_vals = []
        predicted_vals = []
        
        # Changed order to match new glossary: household compositions first, then ethnicity/religion
        for i, row_idx in enumerate(actual_df.index):  # household compositions first
            for j, col_idx in enumerate(actual_df.columns):  # ethnicity/religion categories second
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                # Define threshold for filtering low actual values with no prediction
                threshold = 5
                
                # Filter conditions:
                # 1. Original: both actual and predicted are 0
                # 2. New: actual exists but is below threshold AND predicted is 0 (not predicted)
                should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                
                if not filter_zero_bars or not should_filter:
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
        
        # Calculate R² accuracy using the same method as training
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
        
        # Calculate RMSE
        rmse_val = calculate_rmse(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        rmse_data[idx] = rmse_val
    
    # Calculate number of rows - SIMPLIFIED WITHOUT GEO PLOT
    crosstable_rows = (num_plots + num_cols - 1) // num_cols
    # total_rows = 1 + crosstable_rows  # 1 for geo/legend + crosstable rows - COMMENTED OUT
    total_rows = crosstable_rows  # Only crosstable rows
    
    # Create subplot specifications - SIMPLIFIED WITHOUT GEO PLOT
    specs = []
    
    # # First row: geo plot (left) and legend space (right) - COMMENTED OUT
    # geo_row_specs = []
    # for col in range(num_cols):
    #     if col == 0:
    #         geo_row_specs.append({"type": "geo"})  # Geo plot in first column
    #     else:
    #         geo_row_specs.append(None)  # Empty space for other columns
    # specs.append(geo_row_specs)
    
    # Remaining rows: crosstable plots
    for row in range(crosstable_rows):
        row_specs = []
        for col in range(num_cols):
            row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Row heights - SIMPLIFIED WITHOUT GEO PLOT
    subplot_height = 400 if show_keys else 300
    # geo_row_height = 0.35  # 25% of total height for geo/legend row (increased from 15%) - COMMENTED OUT
    # crosstable_row_height = (1.0 - geo_row_height) / crosstable_rows  # Remaining height divided by crosstable rows - COMMENTED OUT
    
    # row_heights = [geo_row_height] + [crosstable_row_height] * crosstable_rows - COMMENTED OUT
    
    # Create titles - SIMPLIFIED WITHOUT GEO ROW
    # all_titles = [""] * num_cols  # Empty titles for geo row - COMMENTED OUT
    all_titles = []  # Start with empty list
    
    # Add crosstable titles with accuracy and RMSE information
    main_plot_idx = 0
    for i in range(crosstable_rows):
        for j in range(num_cols):
            if main_plot_idx < len(titles):
                accuracy = accuracy_data[main_plot_idx]
                rmse = rmse_data[main_plot_idx]
                all_titles.append(f"{titles[main_plot_idx]} - Acc:{accuracy:.2f}% RMSE:{rmse:.2f}")
                main_plot_idx += 1
            else:
                all_titles.append("")
    
    fig = make_subplots(
        rows=total_rows,
        cols=num_cols,
        subplot_titles=all_titles,
        specs=specs,
        # row_heights=row_heights, - COMMENTED OUT
        vertical_spacing=0.20,  # Significantly increased vertical spacing for better subplot separation
        horizontal_spacing=0.20  # Increased horizontal spacing for better subplot separation
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        # row = (idx // num_cols) + 2  # +2 because first row (index 1) is for geo/legend - COMMENTED OUT
        row = (idx // num_cols) + 1  # +1 for 1-based indexing without geo row
        col = (idx % num_cols) + 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        actual_vals = []
        predicted_vals = []
        category_labels = []  # Store actual category combination labels
        
        # Changed order to match new glossary: household compositions first, then ethnicity/religion
        for i, row_idx in enumerate(actual_df.index):  # household compositions first
            for j, col_idx in enumerate(actual_df.columns):  # ethnicity/religion categories second
                a_val = actual_df.iloc[i, j]
                p_val = predicted_df.iloc[i, j]
                
                # Define threshold for filtering low actual values with no prediction
                threshold = 5
                
                # Filter conditions:
                # 1. Original: both actual and predicted are 0
                # 2. New: actual exists but is below threshold AND predicted is 0 (not predicted)
                should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                
                if not filter_zero_bars or not should_filter:
                    actual_vals.append(a_val)
                    predicted_vals.append(p_val)
                    # Create actual category combination label
                    category_labels.append(f"{row_idx} {col_idx}")
        
        # Use index numbers as x-axis labels instead of actual category labels
        continuous_positions = list(range(1, len(actual_vals) + 1))
        
        # Show index numbers as labels
        visible_labels = [str(i) for i in continuous_positions]
        visible_positions = continuous_positions
        
        # Create bar traces using continuous positions - LEGEND COMMENTED OUT
        actual_trace = go.Bar(
            x=continuous_positions,
            y=actual_vals,
            # name='Actual' if idx == 0 else None, - COMMENTED OUT
            marker_color='red',
            opacity=0.7,
            showlegend=False  # Legend disabled
        )
        
        predicted_trace = go.Bar(
            x=continuous_positions,
            y=predicted_vals,
            # name='Predicted' if idx == 0 else None, - COMMENTED OUT
            marker_color='blue',
            opacity=0.7,
            showlegend=False  # Legend disabled
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update x-axis to show index numbers
        fig.update_xaxes(
            ticktext=visible_labels,
            tickvals=visible_positions,
            tickangle=90,  # 90-degree angle for index numbers
            tickfont=dict(size=10),  # Standard font size for index numbers
            # title_text=titles[idx],  # Add x-axis title as the crosstable name
            row=row,
            col=col
        )
        
        # Update y-axis to show "Number of Households" label
        fig.update_yaxes(
            title_text="Number of Households",
            row=row,
            col=col
        )
    
    # # Add geo plot to the first row, first column - COMMENTED OUT
    # geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    # 
    # if geo_traces and geo_layout:
    #     for trace in geo_traces:
    #         fig.add_trace(trace, row=1, col=1)
    #     
    #     # Update geo subplot layout
    #     fig.update_geos(
    #         geo_layout,
    #         row=1, col=1
    #     )
    #     
    #     # Add area code label below the geo plot
    #     fig.add_annotation(
    #         text=f"Area Code: {selected_area_code}",
    #         xref="paper", yref="paper",
    #         x=0.50, y=0.7,  # Global position centered below the geo plot
    #         xanchor="center", yanchor="top",
    #         showarrow=False,
    #         font=dict(size=12, color="black"),
    #         bgcolor="rgba(255,255,255,0.8)",
    #         bordercolor="black",
    #         borderwidth=1
    #     )
    
    # Update layout with proper sizing - SIMPLIFIED WITHOUT GEO PLOT AND LEGEND
    fig.update_layout(
        height=subplot_height * crosstable_rows,  # Standard height for crosstable rows only
        showlegend=False,  # Legend disabled
        barmode='group',
        plot_bgcolor="white",
        autosize=True,
        margin=dict(
            b=200,  # Increased bottom margin for 90-degree rotated index numbers
            t=80,  # Reduced top margin since no legend needed
            l=60,
            r=60
        ),
        # legend=dict(  # LEGEND CONFIGURATION COMMENTED OUT
        #     orientation="h",  # Horizontal legend
        #     yanchor="top",
        #     y=1.0,  # Position at the top without geo row
        #     xanchor="center", 
        #     x=0.5,  # Centered horizontally
        #     bgcolor='rgba(255,255,255,0.9)'
        # )
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
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Household crosstable comparison plot saved to: {save_path}")
    
    # Display the plot
    fig.show()

# Plotly version of radar crosstable comparison
def plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles, save_path=None):
    """
    Creates radar chart subplots comparing actual vs. predicted distributions for crosstables.
    Uses numeric indices instead of category labels and shows aggregated actual vs predicted lines.
    Now includes a geo plot showing the selected area.
    
    Parameters:
    actual_dfs - Dictionary of crosstable names to actual dataframes
    predicted_dfs - Dictionary of crosstable names to predicted dataframes
    titles - List of subplot titles
    save_path - Optional path to save the plot as HTML
    """
    keys_list = list(actual_dfs.keys())
    num_plots = len(keys_list)
    
    # Pre-calculate accuracy for each crosstable
    accuracy_data = {}
    for idx, crosstable_key in enumerate(keys_list):
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays with new ordering (household compositions first, then ethnicity/religion)
        actual_vals = actual_df.values.flatten()  # No transpose needed - using original order
        predicted_vals = predicted_df.values.flatten()  # No transpose needed - using original order
        
        # Calculate R² accuracy using the same method as training
        r2_accuracy = calculate_r2_accuracy(
            {i: predicted_vals[i] for i in range(len(predicted_vals))},
            {i: actual_vals[i] for i in range(len(actual_vals))}
        )
        accuracy_data[idx] = r2_accuracy * 100.0
    
    # Set to two columns: one for radar charts, one for geo plot
    num_cols = 2
    num_rows = num_plots
    
    # Create subplot specifications
    specs = []
    for row in range(num_rows):
        row_specs = [{'type': 'polar'}]  # Radar chart column
        if row == 0:  # Only add geo to first row
            row_specs.append({'type': 'geo'})  # Geo plot column
        else:
            row_specs.append(None)  # Empty subplot for other rows
        specs.append(row_specs)
    
    # Create complete titles with accuracy information
    extended_titles = []
    for i, title in enumerate(titles):
        accuracy = accuracy_data[i]
        extended_titles.append(f"{title} - Accuracy:{accuracy:.2f}%")
        
        if i == 0:  # Add geo title only for first row
            extended_titles.append("")  # Removed geo plot title
        else:
            extended_titles.append("")  # Empty title for other rows
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=extended_titles,
        specs=specs,
        vertical_spacing=0.1,  # Increased vertical spacing between subplots
        horizontal_spacing=0.2,
        column_widths=[0.7, 0.3]  # 70% radar charts, 30% geo plot
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = idx + 1
        col = 1  # Always put radar charts in first column
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        # Flatten the dataframes to create 1D arrays with new ordering (household compositions first, then ethnicity/religion)
        actual_vals = actual_df.values.flatten()  # No transpose needed - using original order
        predicted_vals = predicted_df.values.flatten()  # No transpose needed - using original order
        
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
        
        # Get pre-calculated accuracy
        r2_accuracy = accuracy_data[idx]
        
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
            name=f'Predicted (Acc: {r2_accuracy:.1f}%)' if idx == 0 else None,
            line=dict(color='blue', width=2),
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
    
    # Add geo plot in top right (first row, second column)
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=2)
        
        # Update geo subplot layout
        fig.update_geos(
            geo_layout,
            row=1, col=2
        )
    
    # Update layout with fixed dimensions
    fig.update_layout(
        height=450 * num_rows,  # Slightly increased height to accommodate title spacing
        width=1200,  # Fixed width
        title_text="Radar Chart Comparison: Actual vs. Predicted",
        title_font_size=18,  # Main title size
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.3,  # Position below geo plot
            xanchor="center", 
            x=0.85,  # Align with geo plot column (for 70/30 layout)
            bgcolor='rgba(255,255,255,0.9)'
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
            col=1  # Only apply to radar chart column
        )
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Household radar chart comparison plot saved to: {save_path}")
    
    # Display the plot
    fig.show()

# Create Plotly individual attribute plots
attribute_dicts = {
    'Composition': (hh_comp_actual, hh_comp_pred),
    'Ethnicity': (ethnicity_actual, ethnicity_pred),
    'Religion': (religion_actual, religion_pred)
}

categories_dict = {
    'Composition': hh_compositions,
    'Ethnicity': ethnicity_categories,
    'Religion': religion_categories
}

household_attributes_save_path = os.path.join(output_dir, 'household_attributes_comparison.html')
plotly_attribute_distributions(attribute_dicts, categories_dict, filter_zero_bars=True, save_path=household_attributes_save_path)

# Create Plotly crosstable plots
actual_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_actual_reshaped,
    'Household_by_Religion': hh_by_religion_actual_reshaped
}

predicted_dfs = {
    'Household_by_Ethnicity': hh_by_ethnicity_pred,
    'Household_by_Religion': hh_by_religion_pred
}

titles = [
    'Household Composition x Ethnicity',
    'Household Composition x Religion'
]

household_crosstable_save_path = os.path.join(output_dir, 'household_crosstable_comparison.html')
plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, show_keys=False, filter_zero_bars=True, save_path=household_crosstable_save_path)

# Create Plotly radar crosstable plots
household_radar_save_path = os.path.join(output_dir, 'household_radar_crosstable_comparison.html')
# plotly_radar_crosstable_comparison(actual_dfs, predicted_dfs, titles, save_path=household_radar_save_path)