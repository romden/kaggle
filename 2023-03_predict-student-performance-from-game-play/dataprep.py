import json, os
from collections import OrderedDict
import pandas as pd
import numpy as np
from config import PATH_INPUT, PATH_WORKING


def load_data(load_feats=True, load_targets=True):

    columns = OrderedDict([
        ('session_id', np.dtype('O')),
        ('level_group', np.dtype('O')), # ['0-4' '5-12' '13-22']
        ('level', np.dtype('int32')),   # from 0 to 22
        
        ('elapsed_time', np.dtype('int32')),
        ('hover_duration', np.dtype('float32')), # 92.4% missing    
        ('room_coor_x', np.dtype('float32')),
        ('room_coor_y', np.dtype('float32')),
        ('screen_coor_x', np.dtype('float32')),
        ('screen_coor_y', np.dtype('float32')),   
        
        #('index', np.dtype('int32')), # index of event (sometimes doesn't correspond to elapsed_time)
        ('page', np.dtype('float32')), # 97.8% missing  [nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ('name', np.dtype('O')),       # nunique = 6    ['basic', 'undefined', 'close', 'open', 'prev', 'next']
        ('event_name', np.dtype('O')), # nunique = 11   ['cutscene_click', 'person_click', ...]        
        ('fqid', np.dtype('O')),       # nunique = 127  ['intro', 'gramps', 'teddy', ...]
        ('room_fqid', np.dtype('O')),  # nunique = 19   ['tunic.historicalsociety.closet', 'tunic.capitol_2.hall', ...]
        ('text_fqid', np.dtype('O')),  # nunique = 126  ['tunic.historicalsociety.closet.intro', ...]
        ('text', np.dtype('O')),       # nunique = 594  ['*COUGH COUGH COUGH*', 'A boring old shirt.', ...]    
        
        #('fullscreen', np.dtype('O')), #np.dtype('float32'), # 100% missing
        #('hq', np.dtype('O')), #np.dtype('float32'), # 100% missing
        #('music', np.dtype('O')), #np.dtype('float32'), # 100% missing
    ])

    if load_feats:
        train = pd.read_csv(f'{PATH_INPUT}/train.csv', dtype=columns, usecols=list(columns.keys()))
    else: 
        train = None

    if load_targets:
        targets = pd.read_csv(f'{PATH_INPUT}/train_labels.csv')
        targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]) )
        targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
    else:
        targets = None

    return train, targets


def save_json(data, filename):    
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)        
    with open(filename, 'w') as f:
        f.write(json.dumps(data, indent=4))

def create_map(s):
    keys = [key.strip() for key in set(s) if not (key is None) and key==key] 
    data = {key: i+1 for i, key in enumerate(sorted(keys))}
    return data

def label_encodings():
    train, targets = load_data(load_feats=True, load_targets=False)
    for col in ['name', 'event_name', 'fqid', 'room_fqid', 'text_fqid', 'text']:
        data = create_map(train[col])
        save_json(data, f'{PATH_WORKING}/model/{col}.json')


def create_features():
    
    from featurizer import feature_engineer

    train, targets = load_data(load_feats=True, load_targets=False)
    
    features = feature_engineer(train) # pd.df with MultiIndex ('session_id', 'level_group')

    filename = f'{PATH_WORKING}/features.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    features.to_csv(filename)


def load_features():
    features = pd.read_csv(f'{PATH_WORKING}/features.csv', dtype={'session_id': str, 'level_group': str})
    features = features.set_index(['session_id', 'level_group'])
    features = features.astype({col: 'float32' for col in features.columns})
    
    targets = pd.read_csv(f'{PATH_INPUT}/train_labels.csv', dtype={'session_id': str, 'correct': 'int8'})
    targets['session'] = targets['session_id'].apply(lambda x: x.split('_')[0])
    targets['question'] = targets['session_id'].apply(lambda x: np.int8(x.split('_')[-1][1:]) )
    targets = targets.set_index(['session', 'question'])
    return features, targets


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other-research/kaggle-2020-02-predict-student-performance-from-game-play/dataprep.py
    """
    label_encodings()


#cp -r /home/datashare/datasets/kaggle/predict-student-performance-from-game-play/input/model /home/BF0772/aapf/other-research/kaggle-2020-02-predict-student-performance-from-game-play/model
