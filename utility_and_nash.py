import numpy as np
import csv

# This assumes player 1 is the attacker and player 2 is the defender

def read_data(filename):
    data = []
    unique_attack_categories = set()  # Create a set to store unique attack categories
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            F = [
                float(row['spkts']), float(row['dpkts']),
                float(row['sbytes']), float(row['dbytes']),
                float(row['rate']), float(row['sttl']),
                float(row['dttl']), float(row['sload']),
                float(row['dload'])
            ]
            Fk = str(row['attack_cat'])  # Convert attack category to string
            unique_attack_categories.add(Fk)  # Add attack category to the set
            data.append((F, Fk))

    return data


def calculate_utility_player1(F, Fk):
    attack_cat_mapping = {'Normal': 0, 'Generic': 1, 'DoS': 2, 'Exploits': 3, 'Worms': 4, 'Analysis': 5, 'Fuzzers': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Backdoor': 9}

    # Convert 'attack_cat' string to numerical value using the mapping
    Fk_numeric = attack_cat_mapping.get(Fk)
    if Fk_numeric is None:
        raise ValueError("Invalid attack category in Fk")
    # Calculate utility using the numerical value
    return np.sum(np.abs(np.array(F) - Fk_numeric))

def calculate_utility_player2(Fk, Fk_prime):
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

def find_nash_equilibrium(data):
    n = len(data)
    min_utility = float('inf')
    nash_equilibrium = None
    print(f"Len of data = {n}")
    count = 0
    for i in range(n):
        count += 1
        print(f"progress {count}")
        for j in range(n):
            F = data[i][0]  # Extract F from data[i]
            Fk = data[j][1]  # Extract Fk from data[j]

            utility_player1 = calculate_utility_player1(F, Fk)
            utility_player2 = calculate_utility_player2(Fk, F)

            if utility_player1 < min_utility and utility_player2 < min_utility:
                min_utility = min(utility_player1, utility_player2)
                nash_equilibrium = (i, j)

    return nash_equilibrium

data_file_path = "./UNSW_NB15_training-set_normalized.csv"
print("reading data")
data = read_data(data_file_path)
print("data consumed\nassessing Nash Equilibrium")
nash_eq = find_nash_equilibrium(data)
print("Nash Equilibrium:", nash_eq)