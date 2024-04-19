import numpy as np

def calculate_utility_player1(F, Fk):
    return np.sum(np.abs(F - Fk))  # Utility function for player 1

def calculate_utility_player2(Fk, Fk_prime):
    return np.sum(np.abs(Fk - Fk_prime))  # Utility function for player 2