import numpy as np
import pandas as pd
import csv
from feature_selection import extract_features

# This assumes player 1 is the attacker and player 2 is the defender

def calculate_utility_player1(F, Fk):
    # TODO Double check this is calculating correctly after feature vector changes
    attack_cat_mapping = {'Normal': 0, 'Generic': 1, 'DoS': 2, 'Exploits': 3, 'Worms': 4, 'Analysis': 5, 'Fuzzers': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Backdoor': 9}

    # Convert 'attack_cat' string to numerical value using the mapping
    Fk_numeric = attack_cat_mapping.get(Fk)
    if Fk_numeric is None:
        raise ValueError("Invalid attack category in Fk")
    # Calculate utility using the numerical value
    return np.sum(np.abs(np.array(F) - Fk_numeric))

def calculate_utility_player2(Fk, Fk_prime):
    # TODO Double check this is calculating correctly after feature vector changes
    attack_cat_mapping = {'Normal': 0, 'Analysis': 1, 'Reconnaissance': 2, 'Shellcode': 3, 'DoS': 4, 'Worms': 5, 'Generic': 6, 'Backdoor': 7, 'Fuzzers': 8, 'Exploits': 9}
    if isinstance(Fk, str) and isinstance(Fk_prime[0], float):
        Fk_numeric = attack_cat_mapping.get(Fk)
        Fk_prime_numeric = Fk_prime[0]
        if Fk_numeric is None:
            raise ValueError("Invalid attack category in Fk")
        return np.sum(np.abs(np.array(Fk_numeric) - np.array(Fk_prime_numeric)))
    else:
        # Fk or Fk_prime is not in the expected format
        raise ValueError("Invalid attack category in Fk or Fk_prime")

def find_nash_equilibrium(feature_vectors_df:pd.DataFrame):
    n = len(feature_vectors_df)
    min_utility_p1 = float('inf')
    min_utility_p2 = float('inf')
    nash_equilibrium = None
    print(f"Len of data = {n}")
    count = 0
    for i in range(n):
        count += 1
        print(f"progress {count}")
        for j in range(n):
            # Extract F (values) from the ith row
            F = feature_vectors_df.iloc[i].values.tolist()
            # Extract Fk (name) from the jth row
            Fk = feature_vectors_df.iloc[j].name

            utility_player1 = calculate_utility_player1(F, Fk)
            utility_player2 = calculate_utility_player2(Fk, F)

            # Check if utility is better for each player at this location 
            if utility_player1 < min_utility_p1 and utility_player2 < min_utility_p2:
                min_utility_p1 = utility_player1
                min_utility_p2 = utility_player2
                nash_equilibrium = (i, j)

    return nash_equilibrium

data_file_path = "./UNSW_NB15_training-set_normalized.csv"
feature_list = ['swin', 'dwin', 'stcpb', 'dtcpb','smean','dmean','trans_depth','response_body_len', 'sinpkt', 'dinpkt','sjit','djit','tcprtt','synack','ackdat']
print("Feature vectors:")
feature_vectors_df = extract_features(data_file_path, feature_list)
print(feature_vectors_df)
# data = read_data(data_file_path)
print("Assessing Nash Equilibrium")
nash_eq = find_nash_equilibrium(feature_vectors_df)
print("Nash Equilibrium:", nash_eq)
