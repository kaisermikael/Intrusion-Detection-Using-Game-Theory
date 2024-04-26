def calculate_recall(TP, FN):
    # TP: True Positives (correctly guessed attack).
    # FN: False Negatives (incorrectly guessed attack).
    return TP / (TP + FN)

def evaluate_recall(predictions, ground_truth):
    recall_rates = {}
    
    # Find unique attack classes
    unique_classes = set(ground_truth)
    
    # Calculate recall for each attack class
    for attack_class in unique_classes:
        # Find indices where the true label is the current attack class
        true_indices = [i for i, label in enumerate(ground_truth) if label == attack_class]
        
        # Count true positives and false negatives
        true_positives = sum(1 for i in true_indices if predictions[i] == attack_class)
        false_negatives = len(true_indices) - true_positives
        
        # Calculate recall rate for the current attack class
        recall = calculate_recall(true_positives, false_negatives)
        
        # Store recall rate in the dictionary
        recall_rates[attack_class] = recall
    
    return recall_rates

# Example usage
# Assuming predictions and ground_truth are lists containing predicted and true labels respectively
recall_rates = evaluate_recall(predictions, ground_truth)
print("Recall rates:", recall_rates)
