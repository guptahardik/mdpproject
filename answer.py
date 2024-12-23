import pandas as pd
import numpy as np
from backward_induction import backward_induction_recursive

def define_MDP(data_file, T, investment_amount, rate, industry):
    # Load the data
    data = pd.read_csv(data_file)

    # Drop the unnamed column
    data = data.drop("Unnamed: 0", axis=1)

    # Ensure the dates are in datetime format
    data['DEAL.DATE'] = pd.to_datetime(data['DEAL.DATE'], format='%Y-%m-%d', errors='coerce')

    # Sort data by TARGET.COMPANY.ID and DEAL.DATE
    data = data.sort_values(by=['TARGET.COMPANY.ID', 'DEAL.DATE'])

    # Define the stages in order
    stages = [
        "Seed", "Series A", "Series B", "Series C", "Series D", "Series E", 
        "Series F", "Series G", "Series H", "Series I", "Series J", "Series K", 
        "Series L", "Pre-IPO", "Venture Debt", "absorption/exit"
    ]
    
    S = len(stages)
    A = 3  # Number of actions: Invest, Hold, Exit
    rterm = np.zeros(S)

    # Initialize a dictionary to store transition matrices for each industry
    transition_matrices = {}
    
    # Calculate transitions for each industry
    for ind, group in data.groupby('PRIMARY.INDUSTRY'):
        # Initialize a transition matrix for the current industry
        transition_matrix = {action: pd.DataFrame(0, index=stages, columns=stages) for action in ['invest', 'hold', 'exit']}

        # Calculate transitions
        for company_id, subgroup in group.groupby('TARGET.COMPANY.ID'):
            previous_stage = None
            previous_date = None
            for index, row in subgroup.iterrows():
                current_stage = row['DEAL.TYPES']
                current_date = row['DEAL.DATE']

                if previous_stage is not None:
                    time_diff = (current_date - previous_date).days / 365.0  # in years
                    #if time_diff <= T:
                    if time_diff <= 1:
                        action = 'invest'
                    else:
                        action = 'hold'

                    transition_matrix[action].loc[previous_stage, current_stage] += 1

                previous_stage = current_stage
                previous_date = current_date

        # Ensure all exit actions have a probability of 1 and no transitions from 'absorption/exit'
        for action in transition_matrix:
            # Set all probabilities for 'absorption/exit' to zero
            transition_matrix[action].loc[:, 'absorption/exit'] = 0
            transition_matrix[action].loc['absorption/exit', :] = 0
            # Set the exit action transition probability to 1
            transition_matrix['exit'].loc[:, 'absorption/exit'] = 1
            transition_matrix['exit'].loc['absorption/exit', 'absorption/exit'] = 1

        # Normalize the transition matrix to get probabilities
        for action in transition_matrix:
            transition_matrix[action] = transition_matrix[action].div(transition_matrix[action].sum(axis=1), axis=0).fillna(0)

        # Store the transition matrix for the current industry
        transition_matrices[ind] = transition_matrix

    # Choose an industry for the MDP definition
    if industry not in transition_matrices:
        raise ValueError(f"Industry '{industry}' not found in the data.")
    
    # Stack the transition matrices along the third dimension for the selected industry
    P = np.zeros((S, S, T, A))
    for action, matrix in transition_matrices[industry].items():
        action_index = ["invest", "hold", "exit"].index(action)
        for t in range(T):
            P[:, :, t, action_index] = matrix.to_numpy()

    # Adjust transition probabilities to enforce "no reinvestment" and "stay in exit" constraints
    for t in range(T):
        for s in range(S):
            P[s, :, t, 0] = 0  # Once you invest, you can't reinvest
            P[s, S-1, t, 2] = 1  # Once you exit, you stay in exit (absorption/exit state)

    def reward(stage, action, valuation):
        if action == "invest":
            return -investment_amount  # Adjusted to encourage investment
        elif action == "hold":
            return -0.5 * rate * investment_amount * (stage + 1)  # Reduced penalty for holding
        elif action == "exit":
            if stage == S - 2:
                return 0
            return 5 * (stage + 1) * valuation  # Adjusted to balance between hold and invest
        return 0

    # Initialize rewards
    r = np.zeros((S, T, A))
    for s, stage in enumerate(stages):
        for t in range(T):
            for a, action in enumerate(["invest", "hold", "exit"]):
                # Calculate reward based on the provided function
                if a == 3:
                    if stage == "venture debt":
                        r[s, t, a] = 0
                    if stage == "absorption/exit":
                        r[s, t, a] = 0
                    r = 10 * (s + 1) - investment_amount
                else:
                    r[s, t, a] = reward(s, action, s + 1) - investment_amount
                    
                  # Stage index starts from 0, so add 1 for valuation

    # Set terminal rewards
    for s, stage in enumerate(stages):
        if stage == "venture debt":
            rterm[s] = 0
        if stage == "absorption/exit":
            rterm[s] = 0
        rterm[s] = 10 * r[s,T-1,2] - investment_amount  # Adjusted to provide significant terminal rewards

    return P, r, rterm

# Call the function with the data file name
P, r, rterm = define_MDP('clean_data.csv', T=4, investment_amount=10000, rate=0.03, industry="Software")

# Use the backward induction method to calculate the policy
_, pi = backward_induction_recursive(P, r, rterm, discount=0.90)

print(pi)
