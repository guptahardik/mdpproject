import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the dataset
file_path = 'preqin_buyout_vc.xlsx'
data = pd.read_excel(file_path)

# Convert relevant columns to appropriate data types
data['DEAL DATE'] = pd.to_datetime(data['DEAL DATE'], errors='coerce')
data['DEAL SIZE (USD MN)'] = pd.to_numeric(data['DEAL SIZE (USD MN)'], errors='coerce')

# Define the rolling window size (e.g., 10 years)
window_size = 30

# Define the end date for rolling windows
end_date = datetime.now()

# Define investment deal types
investment_deal_types = [
    'Seed', 'Series A', 'Series B', 'Series C', 'Series D', 
    'Series E', 'Series F', 'Series G', 'Series H', 'Series I', 
    'Series J', 'Series K', 'Series L', 'Venture Debt', 'PIPE', 
    'Angel', 'Grant', 'Pre-IPO', 'Private Placement/Follow on', 
    'LP Direct', 'GP Stakes', 'Growth', 'Recapitalisation', 
    'Joint Venture'
]

# Define a simplified stage mapping for transition analysis
stage_mapping = {
    'Seed': 0, 'Series A': 1, 'Series B': 2, 'Series C': 3, 'Series D': 4, 
    'Series E': 5, 'Series F': 6, 'Series G': 7, 'Series H': 8, 'Series I': 9, 
    'Series J': 10, 'Series K': 11, 'Series L': 12, 'Unicorn': 13, 'Exit': 14
}

# Filter data to include only investment deal types
data_cleaned = data[data['DEAL TYPES'].isin(investment_deal_types)]

# Map stages to numerical values
data_cleaned['Stage Num'] = data_cleaned['DEAL TYPES'].map(stage_mapping)

# Filter out invalid stages (if any)
data_cleaned = data_cleaned.dropna(subset=['Stage Num'])

# Convert 'Stage Num' to integer type
data_cleaned['Stage Num'] = data_cleaned['Stage Num'].astype(int)

# Sort the data by DEAL DATE and TARGET COMPANY ID
data_sorted = data_cleaned.sort_values(by=['TARGET COMPANY ID', 'DEAL DATE'])

# Function to filter data by a specific time range
def filter_data_by_time_range(data, start_date, end_date):
    return data[(data['DEAL DATE'] >= start_date) & (data['DEAL DATE'] <= end_date)]

# Initialize a dictionary to store results
results = {}

# Iterate through each 10-year window within the range
start_date = data['DEAL DATE'].min()
while start_date + timedelta(days=365*window_size) <= end_date:
    window_end_date = start_date + timedelta(days=365*window_size)
    window_data = filter_data_by_time_range(data_sorted, start_date, window_end_date)
    
    # Initialize dictionaries to track transitions for the current window
    n_states = len(stage_mapping)
    transitions_invest = {industry: np.zeros((n_states, n_states)) for industry in window_data['PRIMARY INDUSTRY'].unique() if industry}
    transitions_hold = {industry: np.zeros((n_states, n_states)) for industry in window_data['PRIMARY INDUSTRY'].unique() if industry}
    
    # Function to count transitions for 'Invest' and 'Hold' actions
    def count_transitions(group):
        industry = group['PRIMARY INDUSTRY'].iloc[0]
        stages = group['Stage Num'].tolist()
        investors = group['INVESTORS'].tolist()
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            current_investors = investors[i]
            next_investors = investors[i + 1]
            if current_stage is not None and next_stage is not None and current_stage < len(stage_mapping) and next_stage < len(stage_mapping):
                # If the same investors are involved, consider it an 'Invest' action
                if current_investors == next_investors and current_stage < next_stage:
                    transitions_invest[industry][current_stage, next_stage] += 1
                # If different investors are involved or the stage does not change, consider it a 'Hold' action
                else:
                    transitions_hold[industry][current_stage, next_stage] += 1
    
    # Group data by TARGET COMPANY ID and apply the count_transitions function
    window_data.groupby('TARGET COMPANY ID').apply(count_transitions)
    
    # Initialize the transition probability matrices for the current window
    P_invest = {industry: np.zeros((n_states, n_states)) for industry in transitions_invest}
    P_hold = {industry: np.zeros((n_states, n_states)) for industry in transitions_hold}
    P_exit = {industry: np.zeros((n_states, n_states)) for industry in transitions_hold}
    
    # Calculate transition probabilities for 'Invest' action
    for industry, matrix in transitions_invest.items():
        for current_stage in range(n_states):
            total_transitions = np.sum(matrix[current_stage, :])
            if total_transitions > 0:
                P_invest[industry][current_stage, :] = matrix[current_stage, :] / total_transitions
    
    # Calculate transition probabilities for 'Hold' action
    for industry, matrix in transitions_hold.items():
        for current_stage in range(n_states):
            total_transitions = np.sum(matrix[current_stage, :])
            if total_transitions > 0:
                P_hold[industry][current_stage, :] = matrix[current_stage, :] / total_transitions
    
    # Define the 'Exit' action transition probabilities
    for industry in P_exit:
        for current_stage in range(n_states):
            if current_stage != stage_mapping['Exit']:
                P_exit[industry][current_stage, stage_mapping['Exit']] = 1.0
            else:
                P_exit[industry][current_stage, stage_mapping['Exit']] = 1.0  # Absorbing state
    
    # Store the transition probabilities for the current window
    results[(start_date, window_end_date)] = {'Invest': P_invest, 'Hold': P_hold, 'Exit': P_exit}
    
    # Move to the next 10-year window
    start_date = start_date + timedelta(days=365)

# Convert results to a DataFrame for easier analysis and export
rows = []
for (start_date, end_date), actions in results.items():
    for action, matrices in actions.items():
        for industry, matrix in matrices.items():
            for current_stage in range(n_states):
                for next_stage in range(n_states):
                    rows.append({
                        'Start Date': start_date,
                        'End Date': end_date,
                        'Primary Industry': industry,
                        'Action': action,
                        'Current Stage': current_stage,
                        'Next Stage': next_stage,
                        'Transition Probability': matrix[current_stage, next_stage]
                    })

transition_prob_df = pd.DataFrame(rows)

# Save the transition probabilities to a CSV file
transition_prob_df.to_csv('transitional_prob.csv', index=False)

print("\nTransition probabilities have been saved to 'transitional_prob.csv'")
