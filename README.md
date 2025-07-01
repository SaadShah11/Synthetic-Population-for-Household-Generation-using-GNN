# Synthetic Population for Household Generation using GNN

This repository contains the implementation of a Graph Neural Network (GNN) based approach for synthetic population generation.
<!-- 
## Requirements

- Python 3.10 or higher
- Dependencies can be installed via pip:
  ```
  pip install -r requirements.txt
  ``` -->

## Project Structure

```
Synthetic Population for Household Generation using GNN/
├── code/
│   ├── generateHouseholds.py       # Household generation logic
│   ├── generateIndividuals.py      # Individual generation logic
│   ├── assignHouseholds.py         # Household assignment logic
│   ├── main.py                     # Main menu interface
│   ├── evaluation.py               # Model evaluation utilities
│   ├── Utils/                      # Utility scripts directory
│   │   ├── createGlossary.py       # Create master glossary file
│   │   ├── plotConvergencePerformance.py  # Generate plots and analysis
│   │   ├── runAssignmentHPTuning.py       # Batch hyperparameter tuning
│   │   └── runMultipleAreas.py            # Batch processing utility
│   └── outputs/                    # Generated outputs directory
├── data/
│   ├── raw-data/                   # Original input data
│   ├── preprocessed-data/          # Processed and prepared data
│   └── encode_data.py              # Data encoding utilities
└── requirements.txt                # Project dependencies
```

## Directory Description

- `code/`: Contains the main implementation files for the GNN-based synthetic population generation
  - `generateHouseholds.py`: Implements the household generation algorithms
  - `generateIndividuals.py`: Implements the individual population generation
  - `assignHouseholds.py`: Contains logic for household assignment and hyperparameter optimization
  - `main.py`: Interactive menu system for running all components
  - `evaluation.py`: Model evaluation and comparison utilities
  - `Utils/`: Utility scripts for batch processing, analysis, and visualization
    - `createGlossary.py`: Creates master glossary files from crosstables
    - `plotConvergencePerformance.py`: Generates convergence and performance plots
    - `runAssignmentHPTuning.py`: Batch hyperparameter tuning for all areas
    - `runMultipleAreas.py`: Batch processing for multiple geographical areas
  - `outputs/`: Stores the generated outputs and results

- `data/`: Houses all data-related files and scripts
  - `raw-data/`: Contains the original, unprocessed input data
  - `preprocessed-data/`: Stores the processed and prepared data for model input
  - `encode_data.py`: Utilities for data encoding and preprocessing

## Getting Started

1 - Ensure you have Python 3.10 or higher installed

2 - Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
3 - Installing Pytorch

GPU:

To install pytorch with cuda support; find you cuda version and install pytorch for that version i.e. for cuda 11.8:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
CPU: 

If you want to use pytorch with CPU then run below command:
   ```
   pip install torch==2.7.0
   ```
4 - Raw and Preprocessed data is in the `data/` directory

5 - Run the main menu interface to access all functionality:
   ```
   cd code
   python main.py
   ```
   
   The main menu provides access to:
   - Individual, household generation and household assignments for specific areas
   - Batch processing for multiple areas
   - Convergence and performance analysis
   - Model evaluation and comparison utilities
