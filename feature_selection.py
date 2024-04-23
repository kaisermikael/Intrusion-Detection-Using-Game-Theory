import pandas as pd

def extract_features(file, feature_list:list):
    feature_list_mean = map(lambda feature: feature + "_mean1234", feature_list)
    data = pd.read_csv("UNSW_NB15_training-set_normalized.csv")

    # Step 1: Separate the data into groups based on attack categories
    attack_groups = data.groupby("attack_cat")

    # Step 2: Extract desired features for each group
    feature_vectors = {}
    for attack_cat, group_data in attack_groups:
        features = group_data[feature_list].mean()
        feature_vectors[attack_cat] = features.values

    # Step 3: Combine the extracted features into feature vectors
    feature_vectors_df = pd.DataFrame.from_dict(feature_vectors, orient='index', columns=feature_list_mean)

    # Return the feature vectors
    return feature_vectors_df