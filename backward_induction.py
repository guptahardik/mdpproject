import numpy as np

def recursive_backward_induction(P, r, rterm, discount, t, S, A, memo, policy):
    """
    Recursive function to compute the optimal policy and value function at each state and time.

    Parameters:
        P: Transition probabilities, shape (S, S, T, A)
        r: Rewards, shape (S, T, A)
        rterm: Terminal rewards, shape (S)
        discount: Discount factor
        t: Current horizon
        S: Number of states
        A: Number of actions
        memo(dict): Memoization dictionary to store computed values
        policy: Optimal policy, shape (S, T)

    Returns:
     Value function for the current stage
    """
    if t in memo:
        return memo[t]
    
    V = np.zeros(S)  # Value function for this stage

    if t == P.shape[2] - 1:  # If at the last stage, use terminal rewards
        for s in range(S):
            Q = np.zeros(A)  # Action values for state s at time t
            for a in range(A):
                # The expected future value from the terminal state rewards
                expected_future_value = np.sum([P[s, sp, t, a] * rterm[sp] for sp in range(S)])
                Q[a] = r[s, t, a] + discount * expected_future_value
            V[s] = np.max(Q)  # Optimal value for state s at time t
            policy[s, t] = np.argmax(Q)  # Optimal action for state s at time t
    else:
        for s in range(S):
            Q = np.zeros(A)  # Action values for state s at time t
            for a in range(A):
                # Calculate expected future values recursively from the next time step's value function
                expected_future_value = np.sum([P[s, sp, t, a] * recursive_backward_induction(P, r, rterm, discount, t + 1, S, A, memo, policy)[sp] for sp in range(S)])
                Q[a] = r[s, t, a] + discount * expected_future_value
            V[s] = np.max(Q)  # Optimal value for state s at time t
            policy[s, t] = np.argmax(Q)  # Optimal action for state s at time t

    memo[t] = V
    return V

def backward_induction_recursive(P, r, rterm, discount):
    """
    Initialize recursion and handle results for the MDP.

    Parameters:
        P: Transition probabilities
        r: Rewards
        rterm: Terminal rewards
        discount: Discount factor
    
    Returns:
        Memoized value functions
        Optimal policy
    """
    S, _, T, A = P.shape
    memo = {}
    policy = np.zeros((S, T), dtype=int)
    recursive_backward_induction(P, r, rterm, discount, 0, S, A, memo, policy)
    return memo, policy
