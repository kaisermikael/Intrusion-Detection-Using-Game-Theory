from utility_and_nash import find_nash_equilibrium
from feature_selection import extract_features

def main():
    data_file_path = "./UNSW_NB15_training-set_normalized.csv"
    feature_list = ['swin', 'dwin', 'stcpb', 'dtcpb','smean','dmean','trans_depth','response_body_len', 'sinpkt', 'dinpkt','sjit','djit','tcprtt','synack','ackdat']

    print("Feature vectors:")
    feature_vectors_df = extract_features(data_file_path, feature_list)
    print(feature_vectors_df)

    print("Assessing Nash Equilibrium")
    nash_eq = find_nash_equilibrium(feature_vectors_df, feature_list)
    print("Nash Equilibrium:", nash_eq)

if __name__ == '__main__':
    main()