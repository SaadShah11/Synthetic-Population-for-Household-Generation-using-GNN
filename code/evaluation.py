#!/usr/bin/env python3
"""
Evaluation script to load and evaluate existing SP_GNN model outputs.
Allows user to select entity type (individuals/households) and area, then generates crosstable plots.
"""

import os
import pandas as pd
import numpy as np
import torch
import json
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import geopandas as gpd
import matplotlib.pyplot as plt

# Define all Oxford areas
ALL_OXFORD_AREAS = [
    'E02005940', 'E02005941', 'E02005942', 'E02005943',
    'E02005944', 'E02005945', 'E02005946', 'E02005947',
    'E02005948', 'E02005949', 'E02005950', 'E02005951',
    'E02005953', 'E02005954', 'E02005955', 'E02005956',
    'E02005957'
]

def create_geo_plot_trace(selected_area_code, current_dir):
    """Create a geo plot trace showing all areas in white except the selected area code which is shaded."""
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
        exclude_codes = ["E02005939", "E02005979", "E02005963", "E02005959"]
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
            colorscale=[[0, "white"], [1, "lightblue"]],
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
        
        # Calculate bounds for the geo layout
        red_bounds = red_bnd.total_bounds
        lon_min, lat_min, lon_max, lat_max = red_bounds
        
        lat_padding = (lat_max - lat_min) * 0.05
        lon_padding = (lon_max - lon_min) * 0.05
        
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

def calculate_r2_accuracy(generated_counts, target_counts):
    """Simple R² measure comparing two distributions"""
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)

    sse = np.sum((gen_vals - tgt_vals) ** 2)
    sst = np.sum((tgt_vals - tgt_vals.mean()) ** 2)

    return 1.0 - sse / sst if sst > 1e-12 else 1.0

def calculate_rmse(generated_counts, target_counts):
    """Calculate RMSE between two distributions"""
    gen_vals = np.array(list(generated_counts.values()), dtype=float)
    tgt_vals = np.array(list(target_counts.values()), dtype=float)
    mse = np.mean((gen_vals - tgt_vals) ** 2)
    return np.sqrt(mse)

def plot_accuracy_over_epochs(convergence_data, output_dir, entity_type="individuals"):
    """
    Plot accuracy over epochs similar to assignHouseholds.py
    Adapted for evaluation script to show training convergence if data is available.
    
    Parameters:
    convergence_data - Dictionary or DataFrame containing epoch-wise accuracy data
    output_dir - Directory to save the plot
    entity_type - Type of entity ("individuals" or "households")
    """
    try:
        # Handle both DataFrame and dictionary inputs
        if isinstance(convergence_data, pd.DataFrame):
            epochs = convergence_data['epochs'].tolist()
            if entity_type == "individuals":
                # For individuals, we can show accuracies for different crosstables
                accuracies_1 = convergence_data.get('accuracies', [])  # Overall accuracy
                accuracies_2 = convergence_data.get('accuracies', [])  # Can be adapted for specific crosstables
                title_1 = "Overall Training Accuracy"
                title_2 = "Overall Training Accuracy"
            else:
                # For households/assignment, show religion and ethnicity accuracies
                accuracies_1 = convergence_data.get('religion_accuracies', [])
                accuracies_2 = convergence_data.get('ethnicity_accuracies', [])
                title_1 = "(a) Religion Accuracy"
                title_2 = "(b) Ethnicity Accuracy"
            
            # Convert pandas Series to list if needed
            if hasattr(accuracies_1, 'tolist'):
                accuracies_1 = accuracies_1.tolist()
            if hasattr(accuracies_2, 'tolist'):
                accuracies_2 = accuracies_2.tolist()
        else:
            # Handle dictionary input
            epochs = convergence_data.get('epochs', [])
            if entity_type == "individuals":
                accuracies_1 = convergence_data.get('accuracies', [])
                accuracies_2 = convergence_data.get('accuracies', [])
                title_1 = "Overall Training Accuracy"
                title_2 = "Overall Training Accuracy"
            else:
                accuracies_1 = convergence_data.get('religion_accuracies', [])
                accuracies_2 = convergence_data.get('ethnicity_accuracies', [])
                title_1 = "(a) Religion Accuracy"
                title_2 = "(b) Ethnicity Accuracy"
        
        # Filter out None values and ensure we have data
        valid_indices = []
        for i, acc in enumerate(accuracies_1):
            if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                valid_indices.append(i)
        
        if not valid_indices:
            print("No valid accuracy data found for plotting.")
            return
        
        epochs_filtered = [epochs[i] for i in valid_indices]
        accuracies_1_filtered = [accuracies_1[i] for i in valid_indices]
        
        # For second accuracy, check if it's different from first
        if accuracies_2 is not None and len(accuracies_2) > 0:
            # Handle pandas Series case
            if hasattr(accuracies_2, 'tolist'):
                accuracies_2 = accuracies_2.tolist()
            valid_indices_2 = []
            for i, acc in enumerate(accuracies_2):
                if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                    valid_indices_2.append(i)
            if valid_indices_2:
                accuracies_2_filtered = [accuracies_2[i] for i in valid_indices_2]
            else:
                accuracies_2_filtered = accuracies_1_filtered  # Fallback
        else:
            accuracies_2_filtered = accuracies_1_filtered
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot first accuracy (e.g., religion or overall)
        # Sample every 10th epoch for cleaner visualization
        step = max(1, len(epochs_filtered) // 20)  # Show up to 20 bars
        epochs_sampled = epochs_filtered[::step]
        acc1_sampled = accuracies_1_filtered[::step]
        
        bars1 = ax1.bar(epochs_sampled, acc1_sampled, color='steelblue', alpha=0.7, 
                       width=max(1, epochs_filtered[-1] / len(epochs_sampled) * 0.8))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(title_1)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels on top of bars (every other bar to avoid crowding)
        for i, (epoch, acc) in enumerate(zip(epochs_sampled, acc1_sampled)):
            if i % 2 == 0:  # Show every other label
                ax1.text(epoch, acc + 1, f'{acc:.1f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Plot second accuracy (e.g., ethnicity or same as first)
        acc2_sampled = accuracies_2_filtered[::step]
        
        bars2 = ax2.bar(epochs_sampled, acc2_sampled, color='steelblue', alpha=0.7,
                       width=max(1, epochs_filtered[-1] / len(epochs_sampled) * 0.8))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(title_2)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on top of bars (every other bar to avoid crowding)
        for i, (epoch, acc) in enumerate(zip(epochs_sampled, acc2_sampled)):
            if i % 2 == 0:  # Show every other label
                ax2.text(epoch, acc + 1, f'{acc:.1f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Display plot
        plt.show()
        print(f"Accuracy over epochs plot displayed for {entity_type}.")
        
    except Exception as e:
        print(f"Error creating accuracy over epochs plot: {e}")
        print("This feature requires convergence data from training runs.")

def load_convergence_data(area_code, current_dir, data_type="individuals"):
    """
    Load convergence data from training runs to create accuracy plots.
    
    Parameters:
    area_code - The area code to load data for
    current_dir - Current directory path
    data_type - Type of data ("individuals", "households", or "assignment")
    
    Returns:
    convergence_data or None if not found
    """
    try:
        if data_type == "individuals":
            convergence_file = os.path.join(current_dir, 'outputs', f'individuals_{area_code}', 'convergence_data.csv')
        elif data_type == "households":
            convergence_file = os.path.join(current_dir, 'outputs', f'households_{area_code}', 'convergence_data.csv')
        elif data_type == "assignment":
            convergence_file = os.path.join(current_dir, 'outputs', f'assignment_hp_tuning_{area_code}', 'best_model_convergence_data.csv')
        else:
            print(f"Unknown data type: {data_type}")
            return None
        
        if os.path.exists(convergence_file):
            convergence_data = pd.read_csv(convergence_file)
            print(f"Loaded convergence data from: {convergence_file}")
            return convergence_data
        else:
            print(f"Convergence data not found at: {convergence_file}")
            return None
            
    except Exception as e:
        print(f"Error loading convergence data: {e}")
        return None

def plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, selected_area_code, current_dir,
                                show_keys=False, num_cols=1, filter_zero_bars=True, save_path=None, entity_type="individuals"):
    """Creates Plotly subplots comparing actual vs. predicted distributions for crosstables."""
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
        
        if entity_type == "individuals":
            # For individuals: age-sex combinations first, then ethnicity/religion/marital
            for j, col_idx in enumerate(actual_df.columns):
                for i, row_idx in enumerate(actual_df.index):
                    a_val = actual_df.iloc[i, j]
                    p_val = predicted_df.iloc[i, j]
                    
                    threshold = 5
                    should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                    
                    if not filter_zero_bars or not should_filter:
                        actual_vals.append(a_val)
                        predicted_vals.append(p_val)
        else:
            # For households: household compositions first, then ethnicity/religion
            for i, row_idx in enumerate(actual_df.index):
                for j, col_idx in enumerate(actual_df.columns):
                    a_val = actual_df.iloc[i, j]
                    p_val = predicted_df.iloc[i, j]
                    
                    threshold = 5
                    should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                    
                    if not filter_zero_bars or not should_filter:
                        actual_vals.append(a_val)
                        predicted_vals.append(p_val)
        
        # Calculate R² accuracy
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
    
    # Calculate number of rows: 1 for geoplot/legend + rows for crosstables
    crosstable_rows = (num_plots + num_cols - 1) // num_cols
    total_rows = 1 + crosstable_rows
    
    # Create subplot specifications
    specs = []
    
    # First row: geo plot (left) and legend space (right)
    geo_row_specs = []
    for col in range(num_cols):
        if col == 0:
            geo_row_specs.append({"type": "geo"})
        else:
            geo_row_specs.append(None)
    specs.append(geo_row_specs)
    
    # Remaining rows: crosstable plots
    for row in range(crosstable_rows):
        row_specs = []
        for col in range(num_cols):
            row_specs.append({"type": "xy"})
        specs.append(row_specs)
    
    # Row heights
    subplot_height = 400 if show_keys else 300
    geo_row_height = 0.25
    crosstable_row_height = (1.0 - geo_row_height) / crosstable_rows
    
    row_heights = [geo_row_height] + [crosstable_row_height] * crosstable_rows
    
    # Create titles
    all_titles = [""] * num_cols  # Empty titles for geo row
    
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
        row_heights=row_heights,
        vertical_spacing=0.25,
        horizontal_spacing=0.10
    )
    
    for idx, crosstable_key in enumerate(keys_list):
        row = (idx // num_cols) + 2
        col = (idx % num_cols) + 1
        
        actual_df = actual_dfs[crosstable_key]
        predicted_df = predicted_dfs[crosstable_key]
        
        actual_vals = []
        predicted_vals = []
        category_labels = []
        
        if entity_type == "individuals":
            # For individuals: age-sex combinations first, then ethnicity/religion/marital
            for j, col_idx in enumerate(actual_df.columns):
                for i, row_idx in enumerate(actual_df.index):
                    a_val = actual_df.iloc[i, j]
                    p_val = predicted_df.iloc[i, j]
                    
                    threshold = 5
                    should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                    
                    if not filter_zero_bars or not should_filter:
                        actual_vals.append(a_val)
                        predicted_vals.append(p_val)
                        category_labels.append(f"{col_idx} {row_idx}")
        else:
            # For households: household compositions first, then ethnicity/religion
            for i, row_idx in enumerate(actual_df.index):
                for j, col_idx in enumerate(actual_df.columns):
                    a_val = actual_df.iloc[i, j]
                    p_val = predicted_df.iloc[i, j]
                    
                    threshold = 5
                    should_filter = (a_val == 0 and p_val == 0) or (0 < a_val < threshold and p_val == 0)
                    
                    if not filter_zero_bars or not should_filter:
                        actual_vals.append(a_val)
                        predicted_vals.append(p_val)
                        category_labels.append(f"{row_idx} {col_idx}")
        
        # Create continuous positions for bars
        continuous_positions = list(range(1, len(actual_vals) + 1))
        visible_labels = [str(label) for label in category_labels]
        visible_positions = continuous_positions
        
        # Create bar traces
        actual_trace = go.Bar(
            x=continuous_positions,
            y=actual_vals,
            name='Actual' if idx == 0 else None,
            marker_color='red',
            opacity=0.7,
            showlegend=idx == 0
        )
        
        predicted_trace = go.Bar(
            x=continuous_positions,
            y=predicted_vals,
            name='Predicted' if idx == 0 else None,
            marker_color='blue',
            opacity=0.7,
            showlegend=idx == 0
        )
        
        fig.add_trace(actual_trace, row=row, col=col)
        fig.add_trace(predicted_trace, row=row, col=col)
        
        # Update x-axis
        fig.update_xaxes(
            ticktext=visible_labels,
            tickvals=visible_positions,
            tickangle=90,
            tickfont=dict(size=12),
            row=row,
            col=col
        )
        
        # Update y-axis
        y_label = "Number of Persons" if entity_type == "individuals" else "Number of Households"
        fig.update_yaxes(
            title_text=y_label,
            row=row,
            col=col
        )
    
    # Add geo plot
    geo_traces, geo_layout = create_geo_plot_trace(selected_area_code, current_dir)
    
    if geo_traces and geo_layout:
        for trace in geo_traces:
            fig.add_trace(trace, row=1, col=1)
        
        fig.update_geos(
            geo_layout,
            row=1, col=1
        )
        
        # Add area code label
        fig.add_annotation(
            text=f"Area Code: {selected_area_code}",
            xref="paper", yref="paper",
            x=0.50, y=0.8,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        height=300 + subplot_height * crosstable_rows,
        showlegend=True,
        barmode='group',
        plot_bgcolor="white",
        autosize=True,
        margin=dict(
            b=400 if show_keys else 300,
            t=100,
            l=60,
            r=60
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.95,
            xanchor="center", 
            x=0.7,
            bgcolor='rgba(255,255,255,0.9)'
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
    
    fig.update_xaxes(
        showline=True,
        linecolor='black',
        linewidth=2
    )
    
    # Save the plot if save_path is provided
    if save_path:
        fig.write_html(save_path)
        print(f"Crosstable evaluation plot saved to: {save_path}")
    
    # Display the plot
    fig.show()

def evaluate_individuals(selected_area_code, current_dir):
    """Evaluate individuals model for the selected area"""
    print(f"\nEvaluating individuals model for area: {selected_area_code}")
    
    # Define paths
    output_dir = os.path.join(current_dir, 'outputs', f'individuals_{selected_area_code}')
    model_file = os.path.join(output_dir, 'person_nodes.pt')
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at {model_file}")
        print("Please run individual generation for this area first.")
        return
    
    print(f"Loading model from: {model_file}")
    
    # Load the model predictions
    person_nodes_tensor = torch.load(model_file, map_location='cpu')
    
    # Extract predictions - format: [age, sex, religion, ethnicity, marital]
    age_pred = person_nodes_tensor[:, 0]
    sex_pred = person_nodes_tensor[:, 1]
    religion_pred = person_nodes_tensor[:, 2]
    ethnicity_pred = person_nodes_tensor[:, 3]
    marital_pred = person_nodes_tensor[:, 4]
    
    # Define categories
    age_groups = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    sex_categories = ['M', 'F']
    ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
    religion_categories = ['C','B','H','J','M','S','O','N','NS']
    marital_categories = ['Single','Married','Partner','Separated','Divorced','Widowed']
    
    # Convert predictions to category names
    sex_pred_names = [sex_categories[i] for i in sex_pred.numpy()]
    age_pred_names = [age_groups[i] for i in age_pred.numpy()]
    ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred.numpy()]
    religion_pred_names = [religion_categories[i] for i in religion_pred.numpy()]
    marital_pred_names = [marital_categories[i] for i in marital_pred.numpy()]
    
    # Load actual data
    ethnic_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/EthnicityBySexByAge.csv'))
    religion_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/ReligionbySexbyAge.csv'))
    marital_by_sex_by_age_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/MaritalbySexbyAgeModified.csv'))
    
    # Filter for selected area
    ethnic_by_sex_by_age_df = ethnic_by_sex_by_age_df[ethnic_by_sex_by_age_df['geography code'] == selected_area_code]
    religion_by_sex_by_age_df = religion_by_sex_by_age_df[religion_by_sex_by_age_df['geography code'] == selected_area_code]
    marital_by_sex_by_age_df = marital_by_sex_by_age_df[marital_by_sex_by_age_df['geography code'] == selected_area_code]
    
    # Create combined age-sex column names
    age_sex_combinations = [f"{age} {sex}" for age in age_groups for sex in sex_categories]
    
    # Create actual crosstables
    ethnic_sex_age_actual = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
    religion_sex_age_actual = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
    marital_sex_age_actual = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)
    
    # Extract actual counts
    for sex in sex_categories:
        for age in age_groups:
            col_name = f"{age} {sex}"
            
            # Ethnicity
            for eth in ethnicity_categories:
                original_col = f'{sex} {age} {eth}'
                if original_col in ethnic_by_sex_by_age_df.columns:
                    ethnic_sex_age_actual.loc[eth, col_name] = ethnic_by_sex_by_age_df[original_col].iloc[0]
            
            # Religion
            for rel in religion_categories:
                original_col = f'{sex} {age} {rel}'
                if original_col in religion_by_sex_by_age_df.columns:
                    religion_sex_age_actual.loc[rel, col_name] = religion_by_sex_by_age_df[original_col].iloc[0]
            
            # Marital
            for mar in marital_categories:
                original_col = f'{sex} {age} {mar}'
                if original_col in marital_by_sex_by_age_df.columns:
                    marital_sex_age_actual.loc[mar, col_name] = marital_by_sex_by_age_df[original_col].iloc[0]
    
    # Create predicted crosstables
    ethnic_sex_age_pred = pd.DataFrame(0, index=ethnicity_categories, columns=age_sex_combinations)
    religion_sex_age_pred = pd.DataFrame(0, index=religion_categories, columns=age_sex_combinations)
    marital_sex_age_pred = pd.DataFrame(0, index=marital_categories, columns=age_sex_combinations)
    
    # Fill predicted crosstables
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
        'Ethnicity x Sex x Age',
        'Religion x Sex x Age',
        'Marital Status x Sex x Age'
    ]
    
    # Plot crosstable comparisons
    save_path = os.path.join(output_dir, 'evaluation_crosstable_comparison.html')
    plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, selected_area_code, current_dir, 
                                 show_keys=False, filter_zero_bars=True, save_path=save_path, entity_type="individuals")
    
    print(f"Individual evaluation plots generated and saved to: {save_path}")

def evaluate_households(selected_area_code, current_dir):
    """Evaluate households model for the selected area"""
    print(f"\nEvaluating households model for area: {selected_area_code}")
    
    # Define paths
    output_dir = os.path.join(current_dir, 'outputs', f'households_{selected_area_code}')
    model_file = os.path.join(output_dir, 'household_nodes.pt')
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at {model_file}")
        print("Please run household generation for this area first.")
        return
    
    print(f"Loading model from: {model_file}")
    
    # Load the model predictions
    household_nodes_tensor = torch.load(model_file, map_location='cpu')
    
    # Extract predictions - format: [household_composition, ethnicity, religion]
    hh_pred = household_nodes_tensor[:, 0]
    ethnicity_pred = household_nodes_tensor[:, 1]
    religion_pred = household_nodes_tensor[:, 2]
    
    # Define categories
    hh_compositions = ['1PE','1PA','1FE','1FM-0C','1FM-2C', '1FM-nA','1FC-0C','1FC-2C','1FC-nA','1FL-nA','1FL-2C','1H-nS','1H-nE','1H-nA', '1H-2C']
    ethnicity_categories = ['W1', 'W2', 'W3', 'W4', 'M1', 'M2', 'M3', 'M4', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'O1', 'O2']
    religion_categories = ['C','B','H','J','M','S','O','N','NS']
    
    # Convert predictions to category names
    hh_comp_pred_names = [hh_compositions[i] for i in hh_pred.numpy()]
    ethnicity_pred_names = [ethnicity_categories[i] for i in ethnicity_pred.numpy()]
    religion_pred_names = [religion_categories[i] for i in religion_pred.numpy()]
    
    # Load actual data
    hhcomp_by_ethnicity_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_ethnicity_Updated.csv'))
    hhcomp_by_religion_df = pd.read_csv(os.path.join(current_dir, '../data/preprocessed-data/crosstables/HH_composition_by_religion_Updated.csv'))
    
    # Filter for selected area
    hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df[hhcomp_by_ethnicity_df['geography code'] == selected_area_code]
    hhcomp_by_religion_df = hhcomp_by_religion_df[hhcomp_by_religion_df['geography code'] == selected_area_code]
    
    # Drop unnecessary columns
    hhcomp_by_ethnicity_df = hhcomp_by_ethnicity_df.drop(columns=['total', 'geography code'])
    hhcomp_by_religion_df = hhcomp_by_religion_df.drop(columns=['total', 'geography code'])
    
    # Create actual crosstables
    hh_by_ethnicity_actual = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
    hh_by_religion_actual = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)
    
    # Extract actual counts
    for hh in hh_compositions:
        for eth in ethnicity_categories:
            col_name = f'{hh} {eth}'
            if col_name in hhcomp_by_ethnicity_df.columns:
                hh_by_ethnicity_actual.loc[hh, eth] = hhcomp_by_ethnicity_df[col_name].iloc[0]
        
        for rel in religion_categories:
            col_name = f'{hh} {rel}'
            if col_name in hhcomp_by_religion_df.columns:
                hh_by_religion_actual.loc[hh, rel] = hhcomp_by_religion_df[col_name].iloc[0]
    
    # Create predicted crosstables
    hh_by_ethnicity_pred = pd.DataFrame(0, index=hh_compositions, columns=ethnicity_categories)
    hh_by_religion_pred = pd.DataFrame(0, index=hh_compositions, columns=religion_categories)
    
    # Fill predicted crosstables
    for i in range(len(hh_comp_pred_names)):
        hh = hh_comp_pred_names[i]
        eth = ethnicity_pred_names[i]
        rel = religion_pred_names[i]
        
        hh_by_ethnicity_pred.loc[hh, eth] += 1
        hh_by_religion_pred.loc[hh, rel] += 1
    
    # Create crosstable dictionaries for plotting
    actual_dfs = {
        'Household_by_Ethnicity': hh_by_ethnicity_actual,
        'Household_by_Religion': hh_by_religion_actual
    }
    
    predicted_dfs = {
        'Household_by_Ethnicity': hh_by_ethnicity_pred,
        'Household_by_Religion': hh_by_religion_pred
    }
    
    titles = [
        'Household Composition x Ethnicity',
        'Household Composition x Religion'
    ]
    
    # Plot crosstable comparisons
    save_path = os.path.join(output_dir, 'evaluation_crosstable_comparison.html')
    plotly_crosstable_comparison(actual_dfs, predicted_dfs, titles, selected_area_code, current_dir, 
                                 show_keys=False, filter_zero_bars=True, save_path=save_path, entity_type="households")
    
    print(f"Household evaluation plots generated and saved to: {save_path}")

def evaluate_assignment(selected_area_code, current_dir):
    """Evaluate assignment model convergence for the selected area"""
    print(f"\nEvaluating assignment model convergence for area: {selected_area_code}")
    
    # Define paths
    output_dir = os.path.join(current_dir, 'outputs', f'assignment_hp_tuning_{selected_area_code}')
    
    # Check if assignment output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Assignment output directory not found at {output_dir}")
        print("Please run household assignment hyperparameter tuning for this area first.")
        return
    
    # Try to load convergence data
    convergence_data = load_convergence_data(selected_area_code, current_dir, "assignment")
    if convergence_data is None:
        print(f"Error: No assignment convergence data found for area {selected_area_code}")
        print("Please ensure assignment hyperparameter tuning has been completed.")
        return
    
    print(f"Loading assignment convergence data from: {output_dir}")
    
    # Create evaluation output directory
    eval_output_dir = os.path.join(current_dir, 'outputs', f'assignment_evaluation_{selected_area_code}')
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Generate accuracy over epochs plot
    plot_accuracy_over_epochs(convergence_data, eval_output_dir, "assignment")
    
    # Print summary statistics from the convergence data
    print(f"\n=== Assignment Training Summary for {selected_area_code} ===")
    if 'religion_accuracies' in convergence_data.columns:
        final_religion_acc = convergence_data['religion_accuracies'].iloc[-1]
        max_religion_acc = convergence_data['religion_accuracies'].max()
        print(f"Final Religion Accuracy: {final_religion_acc:.2f}%")
        print(f"Max Religion Accuracy: {max_religion_acc:.2f}%")
    
    if 'ethnicity_accuracies' in convergence_data.columns:
        final_ethnicity_acc = convergence_data['ethnicity_accuracies'].iloc[-1]
        max_ethnicity_acc = convergence_data['ethnicity_accuracies'].max()
        print(f"Final Ethnicity Accuracy: {final_ethnicity_acc:.2f}%")
        print(f"Max Ethnicity Accuracy: {max_ethnicity_acc:.2f}%")
    
    if 'overall_accuracies' in convergence_data.columns:
        final_overall_acc = convergence_data['overall_accuracies'].iloc[-1]
        max_overall_acc = convergence_data['overall_accuracies'].max()
        print(f"Final Overall Accuracy: {final_overall_acc:.2f}%")
        print(f"Max Overall Accuracy: {max_overall_acc:.2f}%")
    
    if 'losses' in convergence_data.columns:
        final_loss = convergence_data['losses'].iloc[-1]
        min_loss = convergence_data['losses'].min()
        print(f"Final Loss: {final_loss:.6f}")
        print(f"Min Loss: {min_loss:.6f}")
    
    total_epochs = len(convergence_data) if convergence_data is not None else 0
    print(f"Total Training Epochs: {total_epochs}")
    
    # Check for additional files and provide summary
    assignment_files = []
    for filename in ['final_assignments.pt', 'best_hyperparameters.json', 'hp_tuning_results.csv']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            assignment_files.append(filename)
    
    print(f"\nAvailable assignment files: {', '.join(assignment_files)}")
    print(f"Assignment evaluation results saved to: {eval_output_dir}")

def select_entity_type():
    """Allow user to select between individuals, households, and assignment"""
    while True:
        print("\n" + "="*60)
        print("Select Entity Type for Evaluation:")
        print("="*60)
        print("1. Individuals/Persons")
        print("2. Households")
        print("3. Assign Household (Training Convergence)")
        print("="*60)
        
        choice = input("Enter your choice (1-3) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return None
        elif choice == '1':
            return 'individuals'
        elif choice == '2':
            return 'households'
        elif choice == '3':
            return 'assignment'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def select_area():
    """Allow user to select an Oxford area"""
    while True:
        print("\n" + "="*60)
        print("Available Oxford Area Codes:")
        print("="*60)
        
        # Display all areas with indices
        for i, area in enumerate(ALL_OXFORD_AREAS, 1):
            print(f"{i:2d}. {area}")
        
        print(f"\nTotal areas available: {len(ALL_OXFORD_AREAS)}")
        print("="*60)
        
        try:
            choice = input(f"\nSelect area number (1-{len(ALL_OXFORD_AREAS)}) or 'q' to return to main menu: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            area_num = int(choice)
            if 1 <= area_num <= len(ALL_OXFORD_AREAS):
                selected_area = ALL_OXFORD_AREAS[area_num - 1]
                print(f"\nSelected area: {selected_area}")
                return selected_area
            else:
                print(f"Please enter a number between 1 and {len(ALL_OXFORD_AREAS)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def main():
    """Main evaluation function"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Welcome to SP_GNN Model Evaluation!")
    
    # Select entity type
    entity_type = select_entity_type()
    if entity_type is None:
        print("Evaluation cancelled.")
        return
    
    # Select area
    selected_area = select_area()
    if selected_area is None:
        print("Evaluation cancelled.")
        return
    
    # Run evaluation based on entity type
    if entity_type == 'individuals':
        evaluate_individuals(selected_area, current_dir)
    elif entity_type == 'households':
        evaluate_households(selected_area, current_dir)
    else:  # assignment
        evaluate_assignment(selected_area, current_dir)

if __name__ == "__main__":
    main() 