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
Input parameters:
- Investment Amount
- Startup Stage
- Investment Duration
- Industry Preference

Output:
- Optimal policy indicating the best action (Invest, Hold, Exit) for each state over the specified time horizon

## Future Extensions
- Incorporate more sophisticated reward functions
- Analyze additional industries
- Consider external economic factors in investment decisions

## Dependencies
- Python (version X.X)
- [List any additional libraries or frameworks used]

## Data Source
Historical data obtained from Preqin (https://www.preqin.com/)

## Author
Hardik Gupta

## License
[Specify the license under which this project is released]

## Acknowledgements
- Dartmouth Library for providing access to Preqin data
- [Any other acknowledgements or credits]
