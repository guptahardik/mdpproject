# Investment Decision Analysis Using Markov Decision Processes

## Overview
This project applies Markov Decision Processes (MDPs) to analyze investment decisions for startups. It aims to provide venture capitalists and investors with an optimal policy for making investment decisions, maximizing expected returns while minimizing associated risks.

## Motivation
The project addresses the conservative nature of new angel investors in India's economic and venture capital landscape. It explores the application of operations research and financial engineering to assess risks and potential returns in startup investments.

## Goals
- Develop a Markov Decision Process (MDP) model for startup investment decisions
- Maximize expected returns while minimizing associated risks
- Provide investors/VCs with an optimal policy for investment decisions

## Features
- Utilizes historical data from Preqin for analysis
- Considers various startup stages from Seed to Pre-IPO
- Incorporates industry preferences in the decision-making process
- Accounts for investment amount and duration

## Implementation
- Data preparation using Preqin database
- MDP model with states, actions, transition probabilities, and rewards
- Backward induction algorithm for solving the MDP

## Key Components
1. States: Various startup funding stages (e.g., Seed, Series A, Series B, etc.)
2. Actions: Invest, Hold, Exit
3. Transition Probabilities: Calculated from historical data
4. Rewards: Based on investment costs, holding costs, and exit rewards

## Usage

The main script for this project is `answer.py`. This script defines the Markov Decision Process (MDP) model and calculates the optimal investment policy using backward induction.

Input parameters:
- Investment Amount
- Startup Stage
- Investment Duration
- Industry Preference
  

Output:
- Optimal policy indicating the best action (Invest, Hold, Exit) for each state over the specified time horizon

### Key Components of answer.py:
1. **MDP Definition Function**: This function sets up the MDP model using the provided parameters:
- `data_file`: CSV file containing cleaned historical data
- `T`: Time horizon for the investment (in years)
- `investment_amount`: Initial investment amount
- `rate`: Interest rate for calculating holding costs
- `industry`: Specific industry for the investment analysis

2. **Backward Induction**: 
The script uses a recursive backward induction algorithm to solve the MDP and determine the optimal policy.

3. **Output**: 
The script prints the optimal policy matrix, indicating the best action (Invest, Hold, Exit) for each state over the specified time horizon.

### Example Usage
P, r, rterm = define_MDP('clean_data.csv', T=4, investment_amount=10000, rate=0.03, industry="Software")
_, pi = backward_induction_recursive(P, r, rterm, discount=0.90)
print(pi)

This will output the optimal policy matrix for a 4-year investment horizon in the software industry, with an initial investment of $10,000 and a 3% interest rate.


## Future Extensions
- Incorporate more sophisticated reward functions
- Analyze additional industries
- Consider external economic factors in investment decisions

## Dependencies
- Python (version X.X)
- pandas
- numpy

## Data Source
Historical data obtained from Preqin (https://www.preqin.com/)

## Author
Hardik Gupta


## Acknowledgements
- Dartmouth Library for providing access to Preqin data
- ChatGPT-4 for assistance with code cleaning and transitional probability logic
