# Define utility function for player 1 (intruder)
def utility_player1(attack, strategies:dict):
    utilities = {}
    for k1, v1 in strategies.items():
        temp_sum = get_utility(attack, v1)
        for k2, v2 in strategies.items():
            k3 = (k1, k2)
            utilities[k3] = temp_sum
    return utilities

# Define utility function for player 2 (intrusion detection system)
def utility_player2(strategies:dict):
    utilities = {}
    for k1, v1 in strategies.items():
        for k2, v2 in strategies.items():
            temp_sum = get_utility(v1, v2)
            k3 = (k1, k2)
            utilities[k3] = temp_sum
    return utilities

# Define Nash equilibrium function
def find_nash_equilibrium(utilities1, utilities2):
    min_strategies_player1 = find_min_solution(utilities1)
    min_strategies_player2 = find_min_solution(utilities2)
    nash_equilibrium = set(min_strategies_player1) & set(min_strategies_player2)
    return list(nash_equilibrium)

# Function to calculate utility
def get_utility(f1, f2):
    # f1 and f2 are each a feature vector for a chosen strategy
    temp_list = []
    for i, j in zip(f1, f2):
        num = abs(i - j)
        temp_list.append(num)
    return sum(temp_list)

# Function to find minimum solution for a single player
def find_min_solution(utilities:dict):
    min_value = min(utilities.values())
    min_strategies = [k for k, v in utilities.items() if v == min_value]
    return min_strategies
