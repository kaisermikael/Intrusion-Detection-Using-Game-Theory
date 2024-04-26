

import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from feature_selection import get_test_attacks
from utility_and_nash import utility_player1, utility_player2, find_nash_equilibrium

def make_prediction(test_file_path, feature_list, strategies):
    pred_labels = []
    y_labels = []
    attacks = get_test_attacks(test_file_path, feature_list)
    tot_attacks = len(attacks)
    update_interval = tot_attacks // 10  # Update every 10% progress
    for i, attack in enumerate(attacks):
        y_labels.append(attack[-1])

        if i % update_interval == 0:
            progress_percentage = (i / tot_attacks) * 100
            print(f"Progress: {progress_percentage:.0f}%", end="\r")

        utilities_player1 = utility_player1(attack, strategies)
        utilities_player2 = utility_player2(strategies)

        nash_equilibrium = find_nash_equilibrium(utilities_player1, utilities_player2)
        pred_labels.append(nash_equilibrium[0][1])

    print('Progress: 100%')
    return pred_labels, y_labels

def evaluate_prediction(pred_labels, y_labels):
    tp = defaultdict(int) # True Positives
    fn = defaultdict(int) # False Negatives
    for pred, y in zip(pred_labels, y_labels):
        if (pred != 'Normal' and y != 'Normal') or (pred == 'Normal' and y == 'Normal'):
            # Correctly guessed attack was an attack or normal was normal
            tp[y] += 1
        else:
            # Incorrectly guessed normal was an attack or an attack was normal
            fn[y] += 1
    
    recalls = get_recall_rates(tp, fn)
    graph_recalls(recalls)
    return recalls

def get_recall_rates(tp:dict, fn:dict):
    recalls = {}
    for strategy, tp_count in tp.items():
        recalls[strategy] = calc_recall(tp_count, fn[strategy])

    return recalls

def calc_recall(tp, fn):
    return tp / (tp + fn)

def graph_recalls(recalls:dict):
    categories = list(recalls.keys())
    values = list(recalls.values())

    fig, ax = plt.subplots()
    bars = ax.bar(categories, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Recall')
    ax.set_xlabel('Attack Category')
    ax.set_title('Recall by Attack Category')
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels(['{:,.2%}'.format(x) for x in np.arange(0, 1.1, 0.1)])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('recall_graph.png')
