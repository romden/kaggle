import numpy as np
import pandas as pd
import json
from collections import OrderedDict, Counter
from config import PATH_WORKING

CATS = ['page', 'name', 'event_name', 'fqid', 'room_fqid', 'text_fqid'] #+ ['text']
NUMS = ['elapsed_time', 'hover_duration', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y']


def load_encoding():
    data = OrderedDict()    
    for col in CATS:
        if col == 'page':
            encoding = {i-1: i for i in [1, 2, 3, 4, 5, 6, 7]} # for convenience
        else:
            filename = f'{PATH_WORKING}/model/{col}.json'
            encoding = json.load(open(filename, 'r'))
        ordered = OrderedDict()
        for key in sorted(encoding.keys()):
            ordered[key] = np.int32(encoding[key])
        data[col] = ordered       
    return data


class ENCODER:
    data = load_encoding()


def prepare_data(df):

    # sort each group based on elapsed_time
    df = df.sort_values(['session_id', 'elapsed_time'])

    # categorical label encoding
    for col, data in ENCODER.data.items():
        df[col] = df[col].apply(lambda x: data.get(x, 0))

    # dummy variables for computing counts of occurences
    data = OrderedDict()
    for col in CATS:
        for i in ENCODER.data[col].values():
            data[f'{col}_{i}'] = (df[col] == i)
    dummy = pd.DataFrame(data, dtype='int8')
    df = pd.concat([df, dummy], axis=1)

    # differences within groups for consecutive num features
    tmp = df.groupby(['session_id', 'level_group'])[NUMS].diff().add_suffix('_diff')
    # add diff and distances
    df[['elapsed_time_diff', 'hover_duration_diff']] = tmp[['elapsed_time_diff', 'hover_duration_diff']]
    df['room_coor_dist'] = np.linalg.norm(tmp[['room_coor_x_diff', 'room_coor_y_diff']].values, axis=1)
    df['screen_coor_dist'] = np.linalg.norm(tmp[['screen_coor_x_diff', 'screen_coor_y_diff']].values, axis=1)

    return df


def mode(s):
    counter = Counter(s)
    return counter.most_common(1)[0][0] 


def flatten_columns_multi_index(df):
    """flatten multi index for column names"""
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df


def compute_aggregates(grouped, cols, funcs, dtype):
    df = grouped[cols].agg(funcs)
    df = flatten_columns_multi_index(df)
    df = df.astype(dtype)
    return df


def feature_engineer(df):    

    # init
    df = prepare_data(df)
    grouped = df.groupby(['session_id', 'level_group'])    
    dfs = []

    # we compute count once
    cols = CATS[:1]
    funcs = ['count']
    tmp = compute_aggregates(grouped, cols, funcs, 'int32')
    dfs.append(tmp)

    # categorical TODO: k most frequent , pd.Series.mode - can return more than one in a list (gives trouble)
    cols = CATS
    funcs = ['nunique', mode, 'first', 'last']
    tmp = compute_aggregates(grouped, cols, funcs, 'int32')
    dfs.append(tmp)

    # count occurences
    cols = [f'{col}_{i}' for col in CATS for i in ENCODER.data[col].values()]
    funcs = ['sum']
    tmp = compute_aggregates(grouped, cols, funcs, 'int32')
    dfs.append(tmp)

    # numerical
    cols = NUMS
    funcs = ['mean', 'std', 'min', 'max', 'skew', pd.Series.kurtosis]
    tmp = compute_aggregates(grouped, cols, funcs, 'float32')
    dfs.append(tmp)

    # stats for differences in duration and distances
    cols = ['elapsed_time_diff', 'hover_duration_diff', 'room_coor_dist', 'screen_coor_dist']
    funcs = ['mean', 'std', 'min', 'max', 'skew', pd.Series.kurtosis]    
    tmp = compute_aggregates(grouped, cols, funcs, 'float32')
    dfs.append(tmp)
        
    df = pd.concat(dfs, axis=1)
    #df = df.fillna(-1)
    #df = df.reset_index()
    #df = df.set_index('session_id')

    return df


# null values only in hover_duration_ 
