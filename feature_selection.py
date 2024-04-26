import pandas as pd

def extract_features(file, feature_list:list):
    feature_list_mean = map(lambda feature: feature + "_mean", feature_list)
    data = pd.read_csv(file)

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

def get_test_attacks(file, feature_list:list):
    df = pd.read_csv(file)
    return df[feature_list + ['attack_cat']].values.tolist()