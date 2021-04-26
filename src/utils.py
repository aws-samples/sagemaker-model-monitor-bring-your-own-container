import pandas as pd
import numpy as np
import math
import os
import json

from io import BytesIO, StringIO
from bisect import bisect_left
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, scoreatpercentile, chisquare

import boto3

from constants import ColType

le = preprocessing.LabelEncoder()
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


def read_data(filename):
    df = pd.read_csv(filename)
    df.replace('?', np.nan, inplace=True)

    return df


def compute_prob_stats(df_col):
    stats = {
        'nan': df_col.isna().sum()
    }

    if df_col.dtype == 'int64' or df_col.dtype == 'float64':
        stats['type'] = ColType.NUMERICAL
        stats['mean'] = df_col.mean()
        stats['std'] = df_col.std()
        stats['dtype'] = df_col.dtype
    else:
        stats['type'] = ColType.CATEGORICAL

        prob = df_col.value_counts(normalize=True, dropna=False).to_dict()
        stats['prob'] = prob

    return stats


def cum_sum_prob(prob_dict):
    """Calculate cumulative probability from a list of probabilities"""

    if not math.isclose(sum(prob_dict.values()), 1, rel_tol=1e-3):
        ValueError('Input probabilities do not sum to 1.')

    out = []
    cur_sum = 0
    for k, v in prob_dict.items():
        cur_sum += v
        out.append((k, cur_sum))

    return out


def select_item_with_prob(items_prob, n_inst):
    """Select an item random with given discrete pdf"""

    items = []
    for i in range(n_inst):
        pick_prob = np.random.uniform()

        values, probs = zip(*cum_sum_prob(items_prob))
        idx = bisect_left(probs, pick_prob)

        items.append(values[idx])

    return items


def auto_impute_df(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = df[i].fillna(df[i].mode().iloc[0])
            df[i] = le.fit_transform(df[i])
        elif df[i].dtype == 'int' or df[i].dtype == 'float':
            df[i] = df[i].fillna(np.mean(df[i]))

    return df


def create_train_test_split(train_df, target_label):
    train_df = auto_impute_df(train_df)

    x_train, x_test, y_train, y_test = train_test_split(
        train_df.drop(target_label, axis=1), train_df[target_label], test_size=0.1)

    return x_train, x_test, y_train, y_test


def get_column_types(df):
    col_type = {}
    for col in df.columns:
        # If the number of unique items is less than 50 consider them as categorical label
        if df[col].dtype == 'object' or df[col].nunique() < 50:
            col_type[col] = ColType.CATEGORICAL
        else:
            col_type[col] = ColType.NUMERICAL

    return col_type


def normalize(ref_df_col, df_col):
    """Normalize the data using z-score"""

    col_mean = ref_df_col.mean()
    col_std = ref_df_col.std()

    ref_df_norm_col = (ref_df_col - col_mean) / col_std
    df_norm_col = (df_col - col_mean) / col_std

    return ref_df_norm_col, df_norm_col


def get_prob_dist_func(ref_df, df, col_type):
    if col_type == ColType.NUMERICAL:
        # Deciles are constructed to represent probability events
        deciles = [scoreatpercentile(ref_df, i) for i in np.linspace(0, 100, 11)]
        ref_df_prob = pd.cut(ref_df, deciles, duplicates='drop').value_counts(normalize=True).sort_index()
        df_prob = pd.cut(df, deciles, duplicates='drop').value_counts(normalize=True).sort_index()
    else:
        ref_df_prob = ref_df.value_counts(normalize=True, dropna=False)
        ref_df_prob.rename(index={np.nan: 'NaN'}, inplace=True)
        df_prob = df.value_counts(normalize=True, dropna=False)
        df_prob.rename(index={np.nan: 'NaN'}, inplace=True)

    # Find the intersection of labels from both data sets
    common_labels = sorted(set(df_prob.keys()) & set(ref_df_prob.keys()))
    df_filter_prob = [df_prob[k] for k in common_labels]
    ref_df_filter_prob = [ref_df_prob[k] for k in common_labels]

    return ref_df_filter_prob, df_filter_prob, abs(len(df_prob.keys()) - len(ref_df_prob.keys()))


def compute_nan_stats(ref_df, df, col_type):
    df_count = len(df)
    df_nan_count_per = df.isna().sum()/df_count
    ref_df_nan_count_per = ref_df.isna().sum()/len(ref_df)

    nan_diff = abs(df_nan_count_per - ref_df_nan_count_per) * 100

    if col_type == ColType.NUMERICAL:
        p_value_with_nan = ks_2samp(ref_df, df)[1]
        p_value_without_nan = ks_2samp(ref_df.dropna(), df.dropna())[1]

        drift_na = (p_value_with_nan < 0.05) ^ (p_value_without_nan < 0.05)
    elif col_type == ColType.CATEGORICAL:
        ref_df_freq = [(1 - ref_df_nan_count_per) * df_count, ref_df_nan_count_per * df_count]
        df_freq = [(1 - df_nan_count_per) * df_count, df_nan_count_per * df_count]
        drift_na = chisquare(ref_df_freq, df_freq)[1] < 0.05
    else:
        raise ValueError('Column type is neither numerical or categorical')

    return nan_diff, drift_na


def compute_unique_count_drift(df_prob, ref_df_prob):
    """Find the difference in unique counts of two distributions and return as percentage"""

    df_diff = set(df_prob.keys()) - set(ref_df_prob.keys())
    ref_df_diff = set(ref_df_prob.keys()) - set(df_prob.keys())

    return sum([df_prob[k] for k in df_diff] + [ref_df_prob[k] for k in ref_df_diff])


def compute_drift_score(ref_col_prob, col_prob):
    """Compute drift score as the percentage of overlapping probabilities"""

    return sum(abs(np.asarray(ref_col_prob) - np.array(col_prob)) * 100)


def combine_train_infer(train_file, infer_dir):
    """Combine training and inference datasets as one data frame"""

    train_df = pd.read_feather(train_file)

    time_range = range(len([f for f in os.listdir(infer_dir) if 'feather' in f]))
    infer_df_list = [pd.read_feather(f'{infer_dir}/{t}.feather') for t in time_range]

    comb_df_list = []
    train_df.index = [-1] * len(train_df)

    comb_df_list.append(train_df)

    for t in time_range:
        df = infer_df_list[t]
        df.index = [t] * len(df)

        comb_df_list.append(df)

    return pd.concat(comb_df_list), train_df, infer_df_list


def get_bucket_key_from_s3_uri(s3_uri):
    tokens = s3_uri.replace('s3://', '').split('/')
    bucket, s3_key = tokens[0], '/'.join(tokens[1:])
    
    return bucket, s3_key

    
def s3_to_df(s3_uri):
    bucket, s3_key = get_bucket_key_from_s3_uri(s3_uri)
    bucket = s3.Bucket(bucket) 
    
    input_file_io = BytesIO()
    bucket.download_fileobj(s3_key, input_file_io)        

    input_file_io.seek(0)
    
    return pd.read_csv(input_file_io)    


def df_from_datacapture(filename, columns):
    csv_io = StringIO()
    csv_io.writelines(f"{','.join(columns)}\n")
    
    with open(filename, 'r') as f:
        lines = f.readlines()   
        
        for line in lines:
            data = json.loads(line)['captureData']['endpointInput']['data']
            csv_io.write(f'{data}\n')
            
    csv_io.seek(0)
    return pd.read_csv(csv_io)


def construct_df_from_result(s3_uri):
    bucket, s3_key = get_bucket_key_from_s3_uri(s3_uri)
    
    list_resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_key)
    
    df_list = []
    
    # If no data is present
    if 'Contents' not in list_resp:
        return None
    
    for key in list_resp['Contents']:
        obj_resp = s3_client.get_object(Bucket=bucket, Key=key['Key'])  
        json_str = obj_resp['Body'].read().decode('utf-8')
        json_obj = json.loads(json_str)

        df = pd.read_json(json_obj['drift_df'], orient='columns')    
        df['Time'] = [json_obj['end_time'].split(':')[0]] * len(df)
        df['accuracy'] = [json_obj['accuracy']] * len(df)
        df_list.append(df)

    return pd.concat(df_list)
