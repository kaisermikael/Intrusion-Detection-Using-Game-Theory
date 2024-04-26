from evaluate import evaluate_prediction, make_prediction
from feature_selection import extract_features

def main():
    data_file_path = "./UNSW_NB15_training-set_normalized.csv"
    test_file_path = "./UNSW_NB15_testing-set_normalized.csv"
    feature_list = ['swin', 'dwin', 'stcpb', 'dtcpb','smean','dmean','trans_depth','response_body_len', 'sinpkt', 'dinpkt','sjit','djit','tcprtt','synack','ackdat']

    print("Feature vectors:")
    feature_vectors_df = extract_features(data_file_path, feature_list)
    print(feature_vectors_df)
    strategies = {index: row.tolist() for index, row in feature_vectors_df.iterrows()}

    print("\nPredicting...")
    pred_labels, y_labels = make_prediction(test_file_path, feature_list, strategies)
    recalls = evaluate_prediction(pred_labels, y_labels)
    recall_percents = {key: round(value * 100, 2) for key, value in recalls.items()}
    print(f"Recall Rates:\n{recall_percents}")

if __name__ == '__main__':
    main()