import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot convergence and performance data for all areas')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                       help='Output directory containing area folders (default: ../outputs)')
    parser.add_argument('--plot_type', type=str, choices=['individuals', 'households', 'both'], 
                       default='both', help='Type of plots to generate (default: both)')
    return parser.parse_args()

def find_area_folders(output_dir):
    """
    Find all individuals_{area_code} and households_{area_code} folders
    Returns dictionaries with area_code as key and folder path as value
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(current_dir, output_dir)
    
    individuals_folders = {}
    households_folders = {}
    
    if os.path.exists(base_output_dir):
        # Find individuals folders
        individual_pattern = os.path.join(base_output_dir, 'individuals_*')
        for folder in glob.glob(individual_pattern):
            if os.path.isdir(folder):
                area_code = os.path.basename(folder).replace('individuals_', '')
                individuals_folders[area_code] = folder
        
        # Find households folders
        household_pattern = os.path.join(base_output_dir, 'households_*')
        for folder in glob.glob(household_pattern):
            if os.path.isdir(folder):
                area_code = os.path.basename(folder).replace('households_', '')
                households_folders[area_code] = folder
    
    return individuals_folders, households_folders

def load_convergence_data(folder_path):
    """Load convergence data from a folder if it exists"""
    convergence_file = os.path.join(folder_path, 'convergence_data.csv')
    if os.path.exists(convergence_file):
        return pd.read_csv(convergence_file)
    return None

def load_performance_data(folder_path):
    """Load performance data from a folder if it exists"""
    performance_file = os.path.join(folder_path, 'performance_data.csv')
    if os.path.exists(performance_file):
        return pd.read_csv(performance_file)
    return None

def create_convergence_plots(individuals_folders, households_folders, plot_type='both'):
    """Create convergence plots for loss and accuracy"""
    
    # Determine subplot configuration based on plot type
    if plot_type == 'individuals':
        rows, cols = 1, 2
        subplot_titles = [
            'Individuals - Training Loss Convergence',
            'Individuals - Accuracy Convergence'
        ]
    elif plot_type == 'households':
        rows, cols = 1, 2
        subplot_titles = [
            'Households - Training Loss Convergence',
            'Households - Accuracy Convergence'
        ]
    else:  # both
        rows, cols = 2, 2
        subplot_titles = [
            'Individuals - Training Loss Convergence',
            'Individuals - Accuracy Convergence', 
            'Households - Training Loss Convergence',
            'Households - Accuracy Convergence'
        ]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Extended color palette for many area codes
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # Blue, Orange, Green, Red, Purple
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # Brown, Pink, Gray, Olive, Cyan
        '#a6cee3', '#fb9a99', '#fdbf6f', '#cab2d6', '#ff9896',  # Light Blue, Light Red, Light Orange, Light Purple, Light Pink
        '#f0027f', '#386cb0', '#7fc97f', '#beaed4', '#fdc086',  # Magenta, Dark Blue, Light Green, Lavender, Peach
        '#ffff99', '#bf5b17', '#666666'  # Yellow, Dark Orange, Dark Gray
    ]
    
    # Unique marker shapes for every 3 areas to ensure visibility
    marker_shapes = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'star', 'x', 'cross']
    
    # Plot individuals convergence
    color_idx = 0
    if plot_type in ['individuals', 'both'] and individuals_folders:
        for area_code, folder_path in individuals_folders.items():
            convergence_data = load_convergence_data(folder_path)
            if convergence_data is not None:
                color = colors[color_idx % len(colors)]
                marker_shape = marker_shapes[color_idx % len(marker_shapes)]
                
                # Determine row for plotting
                loss_row = 1 if plot_type in ['individuals', 'households'] else 1
                acc_row = 1 if plot_type in ['individuals', 'households'] else 1
                
                # Plot loss with markers every 10 epochs for better visibility
                epochs = convergence_data['epochs']
                losses = convergence_data['losses']
                
                # Sample data points to reduce clutter - take every 50th point for lines (very few values)
                sample_step = 50
                sampled_epochs = epochs[::sample_step]
                sampled_losses = losses[::sample_step]
                
                # Ensure we include the last point if it's not in the sampled data
                if len(epochs) > 0 and len(sampled_epochs) > 0:
                    if epochs.iloc[-1] not in sampled_epochs.values:
                        sampled_epochs = list(sampled_epochs) + [epochs.iloc[-1]]
                        sampled_losses = list(sampled_losses) + [losses.iloc[-1]]
                
                # Sample every 25th epoch for markers to avoid overcrowding
                marker_indices = list(range(0, len(epochs), 25))
                if len(epochs) > 0 and len(epochs) - 1 not in marker_indices:  # Always include the last point
                    marker_indices.append(len(epochs) - 1)
                
                fig.add_trace(
                    go.Scatter(
                        x=sampled_epochs,
                        y=sampled_losses,
                        mode='lines+markers',
                        name=f'Individuals {area_code}',
                        line=dict(color=color, width=3),
                        marker=dict(
                            symbol=marker_shape,
                            size=8,
                            color=color,
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        opacity=0.8,
                        showlegend=True,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Epoch: %{x}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=loss_row, col=1
                )
                
                # Plot accuracy (filter out None values)
                valid_accuracy_data = convergence_data.dropna(subset=['accuracies'])
                if not valid_accuracy_data.empty:
                    acc_epochs = valid_accuracy_data['epochs']
                    accuracies = valid_accuracy_data['accuracies']
                    
                    # Sample data points to reduce clutter - take every 5th point for lines (more values)
                    acc_sample_step = 5
                    sampled_acc_epochs = acc_epochs[::acc_sample_step]
                    sampled_accuracies = accuracies[::acc_sample_step]
                    
                    # Ensure we include the last point if it's not in the sampled data
                    if len(acc_epochs) > 0 and len(sampled_acc_epochs) > 0:
                        if acc_epochs.iloc[-1] not in sampled_acc_epochs.values:
                            sampled_acc_epochs = list(sampled_acc_epochs) + [acc_epochs.iloc[-1]]
                            sampled_accuracies = list(sampled_accuracies) + [accuracies.iloc[-1]]
                    
                    # Sample every 15th epoch for markers (same as households)
                    acc_marker_indices = list(range(0, len(acc_epochs), 15))
                    if len(acc_epochs) > 0 and len(acc_epochs) - 1 not in acc_marker_indices:
                        acc_marker_indices.append(len(acc_epochs) - 1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sampled_acc_epochs,
                            y=sampled_accuracies,
                            mode='lines+markers',
                            name=f'Individuals {area_code}',
                            line=dict(color=color, width=3),
                            marker=dict(
                                symbol=marker_shape,
                                size=8,
                                color=color,
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            opacity=0.8,
                            showlegend=False,  # Already shown in loss plot
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Epoch: %{x}<br>' +
                                        'Accuracy: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        ),
                        row=acc_row, col=2
                    )
                
                color_idx += 1
    
    # Plot households convergence
    if plot_type in ['households', 'both'] and households_folders:
        for area_code, folder_path in households_folders.items():
            convergence_data = load_convergence_data(folder_path)
            if convergence_data is not None:
                color = colors[color_idx % len(colors)]
                marker_shape = marker_shapes[color_idx % len(marker_shapes)]
                
                # Determine row for plotting
                if plot_type == 'households':
                    loss_row, acc_row = 1, 1
                elif plot_type == 'both':
                    loss_row, acc_row = 2, 2
                
                # Plot loss with markers every 10 epochs for better visibility
                epochs = convergence_data['epochs']
                losses = convergence_data['losses']
                
                # Sample data points to reduce clutter - take every 50th point for lines (very few values)
                sample_step = 50
                sampled_epochs = epochs[::sample_step]
                sampled_losses = losses[::sample_step]
                
                # Ensure we include the last point if it's not in the sampled data
                if len(epochs) > 0 and len(sampled_epochs) > 0:
                    if epochs.iloc[-1] not in sampled_epochs.values:
                        sampled_epochs = list(sampled_epochs) + [epochs.iloc[-1]]
                        sampled_losses = list(sampled_losses) + [losses.iloc[-1]]
                
                # Sample every 25th epoch for markers to avoid overcrowding
                marker_indices = list(range(0, len(epochs), 25))
                if len(epochs) > 0 and len(epochs) - 1 not in marker_indices:  # Always include the last point
                    marker_indices.append(len(epochs) - 1)
                
                fig.add_trace(
                    go.Scatter(
                        x=sampled_epochs,
                        y=sampled_losses,
                        mode='lines+markers',
                        name=f'Households {area_code}',
                        line=dict(color=color, width=3),
                        marker=dict(
                            symbol=marker_shape,
                            size=8,
                            color=color,
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        opacity=0.8,
                        showlegend=True,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Epoch: %{x}<br>' +
                                    'Loss: %{y:.4f}<br>' +
                                    '<extra></extra>'
                    ),
                    row=loss_row, col=1
                )
                
                # Plot accuracy (filter out None values)
                valid_accuracy_data = convergence_data.dropna(subset=['accuracies'])
                if not valid_accuracy_data.empty:
                    acc_epochs = valid_accuracy_data['epochs']
                    accuracies = valid_accuracy_data['accuracies']
                    
                    # Sample data points to reduce clutter - take every 5th point for lines (more values)
                    acc_sample_step = 5
                    sampled_acc_epochs = acc_epochs[::acc_sample_step]
                    sampled_accuracies = accuracies[::acc_sample_step]
                    
                    # Ensure we include the last point if it's not in the sampled data
                    if len(acc_epochs) > 0 and len(sampled_acc_epochs) > 0:
                        if acc_epochs.iloc[-1] not in sampled_acc_epochs.values:
                            sampled_acc_epochs = list(sampled_acc_epochs) + [acc_epochs.iloc[-1]]
                            sampled_accuracies = list(sampled_accuracies) + [accuracies.iloc[-1]]
                    
                    # Sample every 15th epoch for markers (same as households)
                    acc_marker_indices = list(range(0, len(acc_epochs), 15))
                    if len(acc_epochs) > 0 and len(acc_epochs) - 1 not in acc_marker_indices:
                        acc_marker_indices.append(len(acc_epochs) - 1)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sampled_acc_epochs,
                            y=sampled_accuracies,
                            mode='lines+markers',
                            name=f'Households {area_code}',
                            line=dict(color=color, width=3),
                            marker=dict(
                                symbol=marker_shape,
                                size=8,
                                color=color,
                                opacity=0.8,
                                line=dict(width=1, color='black')
                            ),
                            opacity=0.8,
                            showlegend=False,  # Already shown in loss plot
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Epoch: %{x}<br>' +
                                        'Accuracy: %{y:.2f}%<br>' +
                                        '<extra></extra>'
                        ),
                        row=acc_row, col=2
                    )
                
                color_idx += 1
    
    # Update axes labels based on plot type
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Training Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    
    if plot_type == 'both':
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Training Loss", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    # Update layout based on plot type
    height = 700 if plot_type in ['individuals', 'households'] else 1200
    title_suffix = plot_type.title() if plot_type != 'both' else 'All Areas'
    
    # Calculate number of legend items for better layout
    total_areas = len(individuals_folders) + len(households_folders)
    legend_cols = min(3, max(1, total_areas // 8))  # Adjust columns based on number of areas
    
    fig.update_layout(
        height=height,
        width=2000,  # Increased width for better legend display
        # title_text=f"Training Convergence Analysis - {title_suffix}",
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),  # Smaller font for more items
            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=80, r=250, t=100, b=80),  # Increased right margin for legend
        plot_bgcolor='white',  # White background for better visibility
        paper_bgcolor='white'
    )
    
    # Style the axes for better visibility - simplified for all plots
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    # Remove extra grid lines and simplify axis styling for accuracy plots
    if plot_type == 'both':
        # Accuracy plots (right column) - cleaner styling
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=2, col=2
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=2, col=2
        )
    elif plot_type == 'individuals':
        # Accuracy plot (right column) - cleaner styling
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
    elif plot_type == 'households':
        # Accuracy plot (right column) - cleaner styling
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            row=1, col=2
        )
    
    return fig

def create_performance_plots(individuals_folders, households_folders, plot_type='both'):
    """Create performance plots showing relationship between population and training efficiency"""
    
    # Determine subplot configuration based on plot type
    if plot_type == 'individuals':
        rows, cols = 2, 1
        subplot_titles = [
            'Individuals - Population vs Avg Epoch Time',
            'Individuals - Population vs Total Training Time'
        ]
    elif plot_type == 'households':
        rows, cols = 2, 1
        subplot_titles = [
            'Households - Population vs Avg Epoch Time',
            'Households - Population vs Total Training Time'
        ]
    else:  # both
        rows, cols = 2, 2
        subplot_titles = [
            'Individuals - Population vs Avg Epoch Time',
            'Households - Population vs Avg Epoch Time',
            'Individuals - Population vs Total Training Time', 
            'Households - Population vs Total Training Time'
        ]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )
    
    # Collect individuals performance data
    individuals_data = []
    for area_code, folder_path in individuals_folders.items():
        performance_data = load_performance_data(folder_path)
        convergence_data = load_convergence_data(folder_path)
        
        if performance_data is not None and convergence_data is not None:
            # Calculate average epoch time if available
            avg_epoch_time = convergence_data['epoch_time_seconds'].mean() if 'epoch_time_seconds' in convergence_data.columns else None
            
            individuals_data.append({
                'area_code': area_code,
                'population': performance_data['num_persons'].iloc[0],
                'total_time_seconds': performance_data['training_time_seconds'].iloc[0],
                'avg_epoch_time': avg_epoch_time,
                'accuracy': performance_data['final_accuracy'].iloc[0],
                'num_epochs': len(convergence_data) if convergence_data is not None else 0
            })
    
    # Collect households performance data
    households_data = []
    for area_code, folder_path in households_folders.items():
        performance_data = load_performance_data(folder_path)
        convergence_data = load_convergence_data(folder_path)
        
        if performance_data is not None and convergence_data is not None:
            # Calculate average epoch time if available
            avg_epoch_time = convergence_data['epoch_time_seconds'].mean() if 'epoch_time_seconds' in convergence_data.columns else None
            
            households_data.append({
                'area_code': area_code,
                'population': performance_data['num_households'].iloc[0],
                'total_time_seconds': performance_data['training_time_seconds'].iloc[0],
                'avg_epoch_time': avg_epoch_time,
                'accuracy': performance_data['final_accuracy'].iloc[0],
                'num_epochs': len(convergence_data) if convergence_data is not None else 0
            })
    
    # Plot individuals - Population vs Average Epoch Time
    if plot_type in ['individuals', 'both'] and individuals_data:
        individuals_df = pd.DataFrame(individuals_data)
        individuals_df = individuals_df.dropna(subset=['avg_epoch_time'])  # Remove rows without epoch time data
        
        if not individuals_df.empty:
            # Calculate efficiency metrics
            individuals_df['persons_per_second'] = individuals_df['population'] / individuals_df['avg_epoch_time']
            
            fig.add_trace(
                go.Scatter(
                    x=individuals_df['population'],
                    y=individuals_df['avg_epoch_time'],
                    mode='markers+text',
                    name='Individuals (Epoch Time)',
                    text=individuals_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=individuals_df['accuracy'],
                        colorscale='Viridis',
                        colorbar=dict(
                            title="Accuracy",
                            x=0.48,
                            len=0.4,
                            y=0.75
                        ),
                        showscale=True,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Avg Epoch Time: %{y:.2f}s<br>' +
                                'Efficiency: %{customdata:.1f} persons/sec<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=individuals_df['persons_per_second']
                ),
                row=1, col=(1 if plot_type == 'individuals' else 1)
            )
            
            # Add trend line for epoch time
            if len(individuals_df) > 1:
                z = np.polyfit(individuals_df['population'], individuals_df['avg_epoch_time'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(individuals_df['population'].min(), individuals_df['population'].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name=f'Trend (slope: {z[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=1, col=(1 if plot_type == 'individuals' else 1)
                )
            
            # Plot individuals - Population vs Total Training Time
            fig.add_trace(
                go.Scatter(
                    x=individuals_df['population'],
                    y=individuals_df['total_time_seconds'],
                    mode='markers+text',
                    name='Individuals (Total Time)',
                    text=individuals_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=individuals_df['accuracy'],
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Total Time: %{y:.1f}s<br>' +
                                'Time per Person: %{customdata:.3f}s<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=individuals_df['total_time_seconds'] / individuals_df['population']
                ),
                row=2, col=(1 if plot_type == 'individuals' else 1)
            )
            
            # Add trend line for total time
            if len(individuals_df) > 1:
                z2 = np.polyfit(individuals_df['population'], individuals_df['total_time_seconds'], 1)
                p2 = np.poly1d(z2)
                y_trend2 = p2(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend2,
                        mode='lines',
                        name=f'Trend (slope: {z2[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=2, col=(1 if plot_type == 'individuals' else 1)
                )
    
        # Plot households - Population vs Average Epoch Time  
    if plot_type in ['households', 'both'] and households_data:
        households_df = pd.DataFrame(households_data)
        households_df = households_df.dropna(subset=['avg_epoch_time'])  # Remove rows without epoch time data
        
        if not households_df.empty:
            # Calculate efficiency metrics
            households_df['households_per_second'] = households_df['population'] / households_df['avg_epoch_time']
            
            fig.add_trace(
                go.Scatter(
                    x=households_df['population'],
                    y=households_df['avg_epoch_time'],
                    mode='markers+text',
                    name='Households (Epoch Time)',
                    text=households_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=households_df['accuracy'],
                        colorscale='Plasma',
                        colorbar=dict(
                            title="Accuracy",
                            x=1.02,
                            len=0.4,
                            y=0.75
                        ),
                        showscale=True,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Avg Epoch Time: %{y:.2f}s<br>' +
                                'Efficiency: %{customdata:.1f} households/sec<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=households_df['households_per_second']
                ),
                row=1, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add trend line for epoch time
            if len(households_df) > 1:
                z3 = np.polyfit(households_df['population'], households_df['avg_epoch_time'], 1)
                p3 = np.poly1d(z3)
                x_trend3 = np.linspace(households_df['population'].min(), households_df['population'].max(), 100)
                y_trend3 = p3(x_trend3)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend3,
                        y=y_trend3,
                        mode='lines',
                        name=f'Trend (slope: {z3[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=1, col=(1 if plot_type == 'households' else 2)
                )
            
            # Plot households - Population vs Total Training Time
            fig.add_trace(
                go.Scatter(
                    x=households_df['population'],
                    y=households_df['total_time_seconds'],
                    mode='markers+text',
                    name='Households (Total Time)',
                    text=households_df['area_code'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=households_df['accuracy'],
                        colorscale='Plasma',
                        showscale=False,
                        line=dict(width=1, color='black')
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Total Time: %{y:.1f}s<br>' +
                                'Time per Household: %{customdata:.3f}s<br>' +
                                'Accuracy: %{marker.color:.3f}<br>' +
                                '<extra></extra>',
                    customdata=households_df['total_time_seconds'] / households_df['population']
                ),
                row=2, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add trend line for total time
            if len(households_df) > 1:
                z4 = np.polyfit(households_df['population'], households_df['total_time_seconds'], 1)
                p4 = np.poly1d(z4)
                y_trend4 = p4(x_trend3)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend3,
                        y=y_trend4,
                        mode='lines',
                        name=f'Trend (slope: {z4[0]:.2e})',
                        line=dict(color='red', dash='dash', width=2),
                        showlegend=False
                    ),
                    row=2, col=(1 if plot_type == 'households' else 2)
                )
    
    # Update axes labels based on plot type
    if plot_type == 'individuals':
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Individuals", row=2, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
    elif plot_type == 'households':
        fig.update_xaxes(title_text="Number of Households", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=2, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
    else:  # both
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=1, col=2)
        fig.update_xaxes(title_text="Number of Individuals", row=2, col=1)
        fig.update_xaxes(title_text="Number of Households", row=2, col=2)
        
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Epoch Time (seconds)", row=1, col=2)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Total Training Time (seconds)", row=2, col=2)
    
    # Update layout based on plot type
    width = 700 if plot_type in ['individuals', 'households'] else 1400
    title_suffix = plot_type.title() if plot_type != 'both' else 'All Types'
    
    fig.update_layout(
        height=800,
        width=width,
        title_text=f"Training Performance vs Population Size Analysis - {title_suffix}",
        title_font_size=18,
        showlegend=False,  # Too many traces, legend would be cluttered
        margin=dict(l=80, r=150, t=100, b=80)
    )
    
    return fig

def create_training_progress_plots(individuals_folders, households_folders, plot_type='both'):
    """Create training progress histograms: Population/Households vs Training Time"""
    
    # Determine subplot configuration based on plot type
    if plot_type == 'individuals':
        rows, cols = 1, 1
        subplot_titles = ['Individuals - Population vs Total Training Time']
    elif plot_type == 'households':
        rows, cols = 1, 1
        subplot_titles = ['Households - Number vs Total Training Time']
    else:  # both
        rows, cols = 1, 2
        subplot_titles = [
            'Individuals - Population vs Total Training Time',
            'Households - Number vs Total Training Time'
        ]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15
    )
    
    # Collect individuals data for histogram
    if plot_type in ['individuals', 'both'] and individuals_folders:
        populations = []
        training_times = []
        area_codes = []
        
        for area_code, folder_path in individuals_folders.items():
            performance_data = load_performance_data(folder_path)
            
            if performance_data is not None:
                populations.append(performance_data['num_persons'].iloc[0])
                training_times.append(performance_data['training_time_seconds'].iloc[0])
                area_codes.append(area_code)
        
        if populations:
            fig.add_trace(
                go.Bar(
                    x=populations,
                    y=training_times,
                    orientation='v',
                    name='Individuals',
                    width=[max(populations) * 0.002] * len(populations),  # Make bars thinner
                    marker=dict(color='red', opacity=0.7),
                    hovertemplate='<b>%{customdata}</b><br>' +
                                'Population: %{x:,}<br>' +
                                'Training Time: %{y:.1f}s<br>' +
                                '<extra></extra>',
                    customdata=area_codes,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add text annotations manually for better control
            for i, (pop, time, code) in enumerate(zip(populations, training_times, area_codes)):
                fig.add_annotation(
                    x=pop,
                    y=time + max(training_times) * 0.15,  # Position further above bar
                    text=code,
                    textangle=90,
                    font=dict(size=8, color='black'),
                    showarrow=False,
                    row=1, col=1
                )
    
    # Collect households data for histogram
    if plot_type in ['households', 'both'] and households_folders:
        num_households = []
        training_times = []
        area_codes = []
        
        for area_code, folder_path in households_folders.items():
            performance_data = load_performance_data(folder_path)
            
            if performance_data is not None:
                num_households.append(performance_data['num_households'].iloc[0])
                training_times.append(performance_data['training_time_seconds'].iloc[0])
                area_codes.append(area_code)
        
        if num_households:
            fig.add_trace(
                go.Bar(
                    x=num_households,
                    y=training_times,
                    orientation='v',
                    name='Households',
                    width=[max(num_households) * 0.002] * len(num_households),  # Make bars thinner
                    marker=dict(color='red', opacity=0.7),
                    hovertemplate='<b>%{customdata}</b><br>' +
                                'Households: %{x:,}<br>' +
                                'Training Time: %{y:.1f}s<br>' +
                                '<extra></extra>',
                    customdata=area_codes,
                    showlegend=False
                ),
                row=1, col=(1 if plot_type == 'households' else 2)
            )
            
            # Add text annotations manually for better control
            for i, (hh, time, code) in enumerate(zip(num_households, training_times, area_codes)):
                fig.add_annotation(
                    x=hh,
                    y=time + max(training_times) * 0.15,  # Position further above bar
                    text=code,
                    textangle=90,
                    font=dict(size=8, color='black'),
                    showarrow=False,
                    row=1, col=(1 if plot_type == 'households' else 2)
                )
    
    # Update axes based on plot type
    if plot_type == 'individuals':
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
    elif plot_type == 'households':
        fig.update_xaxes(title_text="Number of Households", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
    else:  # both
        fig.update_xaxes(title_text="Number of Individuals", row=1, col=1)
        fig.update_xaxes(title_text="Number of Households", row=1, col=2)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="Training Time (seconds)", row=1, col=2)
    

    
    # Update layout based on plot type
    width = 2300 if plot_type in ['individuals', 'households'] else 2600
    title_suffix = plot_type.title() if plot_type != 'both' else 'All Areas'
    
    fig.update_layout(
        height=600,
        width=width,
        title_text=f"Population vs Training Time Analysis - {title_suffix}",
        title_font_size=18,
        showlegend=False,
        margin=dict(l=80, r=150, t=100, b=80),
        plot_bgcolor="white"
    )
    
    # Style the axes
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
    
    return fig

def main():
    args = parse_arguments()
    
    print("Searching for existing output folders...")
    individuals_folders, households_folders = find_area_folders(args.output_dir)
    
    print(f"Found {len(individuals_folders)} individuals folders:")
    for area_code in individuals_folders.keys():
        print(f"  - individuals_{area_code}")
    
    print(f"Found {len(households_folders)} households folders:")
    for area_code in households_folders.keys():
        print(f"  - households_{area_code}")
    
    if not individuals_folders and not households_folders:
        print("No output folders found. Please run generateIndividuals.py and/or generateHouseholds.py first.")
        return
    
    # Create convergence plots
    if individuals_folders or households_folders:
        print(f"\nCreating convergence plots for {args.plot_type}...")
        convergence_fig = create_convergence_plots(individuals_folders, households_folders, args.plot_type)
        convergence_fig.show()
        
        # Save convergence plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plot_type_suffix = f"_{args.plot_type}" if args.plot_type != 'both' else ""
        convergence_plot_path = os.path.join(current_dir, args.output_dir, f'convergence_plots{plot_type_suffix}.html')
        os.makedirs(os.path.dirname(convergence_plot_path), exist_ok=True)
        convergence_fig.write_html(convergence_plot_path)
        print(f"Convergence plots saved to: {convergence_plot_path}")
    
    # Create performance plots
    # if individuals_folders or households_folders:
        # print(f"\nCreating performance plots for {args.plot_type}...")
        # performance_fig = create_performance_plots(individuals_folders, households_folders, args.plot_type)
        # performance_fig.show()
        
        # Save performance plot
        # performance_plot_path = os.path.join(current_dir, args.output_dir, f'performance_plots{plot_type_suffix}.html')
        # performance_fig.write_html(performance_plot_path)
        # print(f"Performance plots saved to: {performance_plot_path}")
    
    # Create training progress plots
    # if individuals_folders or households_folders:
    #     print(f"\nCreating training progress plots for {args.plot_type}...")
    #     training_progress_fig = create_training_progress_plots(individuals_folders, households_folders, args.plot_type)
    #     training_progress_fig.show()
        
    #     # Save training progress plot
    #     training_progress_plot_path = os.path.join(current_dir, args.output_dir, f'training_progress_plots{plot_type_suffix}.html')
    #     training_progress_fig.write_html(training_progress_plot_path)
    #     print(f"Training progress plots saved to: {training_progress_plot_path}")
    
    # Create summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    if individuals_folders:
        print("\nIndividuals:")
        for area_code, folder_path in individuals_folders.items():
            performance_data = load_performance_data(folder_path)
            convergence_data = load_convergence_data(folder_path)
            if performance_data is not None:
                pop = performance_data['num_persons'].iloc[0]
                time_sec = performance_data['training_time_seconds'].iloc[0]
                accuracy = performance_data['final_accuracy'].iloc[0]
                
                # Add epoch timing info if available
                avg_epoch_time = "N/A"
                if convergence_data is not None and 'epoch_time_seconds' in convergence_data.columns:
                    avg_epoch_time = f"{convergence_data['epoch_time_seconds'].mean():.2f}s"
                
                print(f"  {area_code}: {pop:,} persons, {time_sec:.1f}s total, {avg_epoch_time} avg/epoch, {accuracy:.3f} accuracy")
    
    if households_folders:
        print("\nHouseholds:")
        for area_code, folder_path in households_folders.items():
            performance_data = load_performance_data(folder_path)
            convergence_data = load_convergence_data(folder_path)
            if performance_data is not None:
                hh = performance_data['num_households'].iloc[0]
                time_sec = performance_data['training_time_seconds'].iloc[0]
                accuracy = performance_data['final_accuracy'].iloc[0]
                
                # Add epoch timing info if available
                avg_epoch_time = "N/A"
                if convergence_data is not None and 'epoch_time_seconds' in convergence_data.columns:
                    avg_epoch_time = f"{convergence_data['epoch_time_seconds'].mean():.2f}s"
                
                print(f"  {area_code}: {hh:,} households, {time_sec:.1f}s total, {avg_epoch_time} avg/epoch, {accuracy:.3f} accuracy")

if __name__ == "__main__":
    main() 