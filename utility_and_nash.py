import numpy as np
import csv

def read_data(data_file_path):
    data = []
    with open(data_file_path, 'r') as file:
        reader = csv.DictReaderFile(file)
        for row in reader:
            F = [float(row[''])]
            Fk = [float(row[''])]
            data.append((F, Fk))
    return data
    

def calculate_utility_player1(F, Fk):
    return np.sum(np.abs(F - Fk))  # Utility function for player 1

def calculate_utility_player2(Fk, Fk_prime):
    return np.sum(np.abs(Fk - Fk_prime))  # Utility function for player 2

def find_nash_equilibrium(data):
    n = len(data)
    min_utility = float('inf')
    nash_equilibrium = None

    for i in range(n):
        for j in range(n):
            F = data[i]
            Fk = data[j]

            utility_player1 = calculate_utility_player1(F, Fk)
            utility_player2 = calculate_utility_player2(Fk, Fk_prime)

            if utility_player1 < min_utility and utility_player2 < min_utility:
                min_utility = min(utility_player1, utility_player2)
                nash_equilibrium = (i, j)

    return nash_equilibrium


data_file_path = "./UNSW_NB15_training-set_normalized.csv"
data = read_data()
nash_eq = find_nash_equilibrium(data)
print("Nash Equilibrium:", nash_eq)