# Detecting Covariate Shift within Data

import os
import json 
import argparse
import pathlib

import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance, ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest

import utils

from constants import ColType


def compute_drift(train_df, infer_df):
    col_type = utils.get_column_types(train_df)

    train_count, infer_count = len(train_df), len(infer_df)

    drift_data = []

    for col in train_df.columns:
        drift_col = {'Feature': col, 'Column Type': col_type[col]}

        train_col, infer_col = train_df[col], infer_df[col]
        train_prob, infer_prob, compute_unique_count_drift = \
            utils.get_prob_dist_func(train_col, infer_col, col_type[col])

        drift_col['drift_score'] = utils.compute_drift_score(train_prob, infer_prob)

        drift_col['NaN % Diff'], drift_col['is_nan_signif'] = \
            utils.compute_nan_stats(train_col, infer_col, col_type[col])

        if col_type[col] == ColType.NUMERICAL:
            train_norm_col, infer_norm_col = utils.normalize(train_col, infer_col)
            drift_col['KS'], drift_col['p-value'] = ks_2samp(train_norm_col, infer_norm_col)
            drift_col['wasserstein_distance'] = wasserstein_distance(train_norm_col, infer_norm_col)
        elif col_type[col] == ColType.CATEGORICAL:
            # Chisquare requires frequency and it has to be larger than 5. Hence it's multiplied by the size of
            # the inference data set.
            train_freq = [int(p * infer_count) for p in train_prob]
            infer_freq = [int(p * infer_count) for p in infer_prob]
            drift_col['chisquare'], drift_col['p-value'] = chisquare(train_freq, infer_freq)

            drift_col['Unique Count Drift'] = compute_unique_count_drift

        drift_col['jensenshannon'] = jensenshannon(train_prob, infer_prob)

        drift_data.append(drift_col)

    drift_df = pd.DataFrame(drift_data)

    return drift_df


def compute_drift_multiple_inst(train_file, infer_dir):
    """Compute drift for inference datasets"""

    train_df = pd.read_feather(train_file)

    time_range = range(len([f for f in os.listdir(infer_dir) if 'feather' in f]))
    infer_df_list = [pd.read_feather(f'{infer_dir}/{t}.feather') for t in time_range]

    drift_df_list = []
    for t in time_range:
        df = compute_drift(train_df, infer_df_list[t])

        df.index = [t + 1] * len(df)
        drift_df_list.append(df)

    return pd.concat(drift_df_list)


def extract_significant_drifts(drift_df):
    """Determine significant feature drifts and separate them into numerical and categorical"""

    numerical_feat_drift = set()
    cat_feat_drift = set()

    drift_signi_df = drift_df[drift_df['p-value'] < 0.05]

    numerical_feat_drift.update(drift_signi_df[drift_signi_df['Column Type'] == ColType.NUMERICAL].Feature.tolist())
    cat_feat_drift.update(drift_signi_df[drift_signi_df['Column Type'] == ColType.CATEGORICAL].Feature.tolist())

    return numerical_feat_drift, cat_feat_drift


def compute_accuracy_with_drift(test_df, infer_df, target_label):
    """Compute accuracy loss due to model drift."""
    test_df = utils.auto_impute_df(test_df)
    
    x_pos = test_df[test_df[target_label] == 1].drop(target_label, axis=1)

    # Create Isolation Forest to create one-class classifier to identify positive labels
    isof = IsolationForest(n_estimators=100, n_jobs=-1, contamination=0, verbose=1, max_features=x_pos.shape[1],
                           max_samples=x_pos.shape[0], bootstrap=False, random_state=123)

    isof.fit(x_pos)
    scores = isof.score_samples(x_pos)
    pos_score_mean = np.mean(scores)

    # Compute anomaly scores of inference data and compare with original scores computed on positive samples of test
    # dataset
    infer_df = utils.auto_impute_df(infer_df)
    pred_score_mean = np.mean(isof.score_samples(infer_df))

    accuracy = 100 - abs((pred_score_mean - pos_score_mean) / pos_score_mean) * 100

    return accuracy


def main(train_s3_uri, test_s3_uri, target_label):    
    train_df = utils.s3_to_df(train_s3_uri)
    train_df.drop([target_label], axis=1, inplace=True)
    
    test_df = utils.s3_to_df(test_s3_uri)
    
    infer_dir = os.environ['dataset_source']
    
    infer_df_list = []
    for filepath in pathlib.Path(infer_dir).rglob('*.jsonl'):
        print(filepath)
        df = utils.df_from_datacapture(filepath.absolute(), train_df.columns.to_list())
        infer_df_list.append(df)
        
    infer_df = pd.concat(infer_df_list)
    
    drift_df = compute_drift(train_df, infer_df)
    accuracy = compute_accuracy_with_drift(test_df, infer_df, target_label)
    
    output = {
        'accuracy': accuracy, 
        'drift_df': drift_df.to_json(),
        'end_time': os.environ['end_time']
    }

    with open(f"{os.environ['output_path']}/results.json", 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_s3_uri', help='Training dataset S3 URL')
    parser.add_argument('--test_s3_uri', help='Testing dataset S3 URL')
    parser.add_argument('--target_label', help='Target label')
    
    args = parser.parse_args()
    
    main(train_s3_uri=args.train_s3_uri, test_s3_uri=args.test_s3_uri, target_label=args.target_label)
