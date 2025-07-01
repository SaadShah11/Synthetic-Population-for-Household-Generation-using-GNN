import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Age categories
child_Ages = ["0_4", "5_7", "8_9", "10_14", "15"]
adult_Ages = ["16_17", "18_19", "20_24", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_64"]
elder_Ages = ["65_69", "70_74", "75_79", "80_84", "85+"]


# Function to extract random sample and update persons dataframe
def extract_random_sample(df, persons, n):
    if n <= 0:
        return [], persons

    if len(df) >= n:
        sampled = df.sample(n)
        sampled_PIDs = sampled['PID'].tolist()
        persons = persons[~persons['PID'].isin(sampled_PIDs)]
        return sampled_PIDs, persons
    else:
        sampled_PIDs = df['PID'].tolist()
        persons = persons[~persons['PID'].isin(sampled_PIDs)]
        return sampled_PIDs, persons


# Function to assign persons to households
def assign_households(households, persons):
    household_assignments = []

    for _, household in households.iterrows():
        composition = household['Composition']
        size = household['Size']

        sampled_PIDs, persons = assign_based_on_composition(composition, size, persons)

        household_assignments.append(sampled_PIDs)

        if len(sampled_PIDs) < size:
            print(f"Warning: Household {household['HID']} could not be fully filled. Assigned {len(sampled_PIDs)} members out of {size}.")

    households['Assigned_PIDs'] = household_assignments
    return households


# Assign based on the composition rules
def assign_based_on_composition(composition, size, persons):
    if persons.empty or size <= 0:
        return [], persons

    if composition == '1PE' or composition == '1FE':
        elders = persons[persons['Age'].isin(elder_Ages)]
        return extract_random_sample(elders, persons, size)

    elif composition == '1PA':
        adults = persons[persons['Age'].isin(adult_Ages)]
        return extract_random_sample(adults, persons, size)

    elif composition.startswith('1FM'):
        married_male = persons[(persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'M')]
        married_female = persons[(persons['MaritalStatus'] == 'Married') & (persons['Sex'] == 'F')]
        pids_male, persons = extract_random_sample(married_male, persons, 1)
        pids_female, persons = extract_random_sample(married_female, persons, 1)
        remaining_size = max(0, size - 2)
        if '2C' in composition:
            children = persons[persons['Age'].isin(child_Ages)]
            pids_children, persons = extract_random_sample(children, persons, remaining_size)
            return pids_male + pids_female + pids_children, persons
        else:
            return pids_male + pids_female, persons

    return [], persons


# Function to verify household assignments
def verify_households(households, persons):
    correct_assignments = 0
    total_households = len(households)
    composition_errors = {}

    size_errors_total = 0

    for _, household in households.iterrows():
        assigned_pids = household['Assigned_PIDs']
        size = household['Size']
        composition = household['Composition']

        # Initialize composition error count if not already present
        if composition not in composition_errors:
            composition_errors[composition] = 0

        # Get assigned persons from the persons dataframe
        assigned_persons = persons[persons['PID'].isin(assigned_pids)]

        # Rule 1: Check that assigned persons <= size
        if len(assigned_persons) > size:
            size_errors_total += 1
            continue

        # Rule 2: Verify composition logic
        if not verify_composition_logic(composition, assigned_persons):
            composition_errors[composition] += 1
            continue

        correct_assignments += 1

    return size_errors_total, composition_errors, correct_assignments


# Helper function to verify the composition logic
def verify_composition_logic(composition, assigned_persons):
    if composition == '1PE' or composition == '1FE':
        return all(assigned_persons['Age'].isin(elder_Ages)) and len(assigned_persons) == 1

    elif composition == '1PA':
        return all(assigned_persons['Age'].isin(adult_Ages)) and len(assigned_persons) == 1

    elif composition.startswith('1FM'):
        married_male = assigned_persons[(assigned_persons['MaritalStatus'] == 'Married') & (assigned_persons['Sex'] == 'M')]
        married_female = assigned_persons[(assigned_persons['MaritalStatus'] == 'Married') & (assigned_persons['Sex'] == 'F')]
        if '2C' in composition:
            children = assigned_persons[assigned_persons['Age'].isin(child_Ages)]
            return len(married_male) == 1 and len(married_female) == 1 and len(children) == len(assigned_persons) - 2
        else:
            return len(married_male) == 1 and len(married_female) == 1

    return True


# Plot the verification graph for errors
def plot_error_graph(size_errors_total, composition_errors):
    compositions = list(composition_errors.keys())
    composition_logic_errors = [composition_errors[c] for c in compositions]

    # Create bar graph
    plt.figure(figsize=(12, 6))

    # Plotting size constraint errors as a single bar
    plt.bar(['Size Constraint'], [size_errors_total], color='lightcoral', label='Size Constraint Errors')

    # Plotting composition logic errors
    plt.bar(compositions, composition_logic_errors, color='skyblue', label='Composition Logic Errors')

    # Adding labels, title, and legend
    plt.xlabel('Household Composition')
    plt.ylabel('Number of Errors')
    plt.title('Errors by Household Composition')
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# Load the data
persons = pd.read_csv('generated_population2.csv')
persons['PID'] = range(1, len(persons) + 1)

households = pd.read_csv('generated_households.csv')
households['HID'] = range(1, len(households) + 1)

# Assign persons to households
households = assign_households(households, persons)

# Verify households and calculate accuracy
size_errors_total, composition_errors, correct_assignments = verify_households(households, persons)
print(correct_assignments/len(households)*100)
print(size_errors_total + sum(composition_errors.values()))
# Plot the verification results for errors
# plot_error_graph(size_errors_total, composition_errors)
