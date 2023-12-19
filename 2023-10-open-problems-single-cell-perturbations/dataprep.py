import numpy as np
import pandas as pd
import os
import pickle
import itertools

from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

from shared import PATH_DATA


def load_train_df():
    filename = os.path.join(PATH_DATA, 'input', 'de_train.parquet') 
    de_train_df = pd.read_parquet(filename)

    columns = ['cell_type', 'sm_name'] + de_train_df.loc[:,'A1BG':].columns.tolist()
    train_df = de_train_df.loc[~de_train_df['control'],columns]

    encoder = pickle.load(open(os.path.join(PATH_DATA, 'data', 'encoder.pkl'), 'rb'))
    train_df[['cell_type', 'sm_name']] = encoder.transform(train_df[['cell_type', 'sm_name']] .values)
    return train_df


def add_predictions(train_df):     

    # submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.528_nn.csv') )
    # id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  ) 
    # id_map_df[['cell_type', 'sm_name']] = np.load(os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy'))
    # test_df = id_map_df.merge(submission_df, on='id', how='left').drop(columns='id')
    
    data = np.load(os.path.join(PATH_DATA, 'pivot_1_booster_6+64', 'xx+predictions.npy'))
    test_df = pd.DataFrame(data, columns=train_df.columns)
    test_df = test_df.merge(train_df, on=['cell_type', 'sm_name'], how='left', indicator=True, suffixes=(None, '_right'))
    test_df = test_df.query('_merge == "left_only"')[train_df.columns]

    xx = [[i, j] for i in range(6) for j in range(144)]    
    df = pd.DataFrame(xx, columns=['cell_type', 'sm_name'])

    all_df = pd.concat([train_df, test_df], axis=0)
    df = df.merge(all_df, on=['cell_type', 'sm_name'], how='left')
    df.loc[:,'A1BG':] = SimpleImputer().fit_transform(df.loc[:,'A1BG':].values)

    assert len(df) == len(df.drop_duplicates(subset=['cell_type', 'sm_name'])) 
    assert len(df) == 6*144
    
    return df


def create_targets(folder):
    
    train_df = load_train_df()
    train_df = add_predictions(train_df) 
    
    n_genes = 18211
    n_sm_names = 144
    n_cell_types = 6
 
    targets = np.zeros((n_cell_types*n_genes, n_sm_names), 'float32')    
    
    for i, gene in enumerate(train_df.loc[:,'A1BG':].columns):
        tmp_df = pd.pivot_table(train_df, values=gene, index=['cell_type'], columns=['sm_name'])
        head = i*n_cell_types
        tail = (i+1)*n_cell_types
        targets[head:tail,:] = tmp_df.values
        
    np.save(os.path.join(folder, 'targets_on_booster_6+64.npy'), targets)


def create_embedding():

    train_df = load_train_df()

    values = train_df.drop(columns='sm_name').groupby('cell_type').mean().values
    cell_type = TruncatedSVD(n_components=6, n_iter=7).fit_transform(values)
    
    values = train_df.loc[:,'A1BG':].values.T
    gene = TruncatedSVD(n_components=122, n_iter=7).fit_transform(values)

    return cell_type, gene


def create_features(folder):    

    # ordinal encodings
    data = []
    for tpl in itertools.product(range(18211), range(6)):
        data.append(list(tpl))
    
    xx = np.array(data, dtype='int32')
    xx = xx[:,[1,0]] # cell - 0 column, gene - 1 column

    # features
    cell_type, gene = create_embedding()
    n, m = cell_type.shape[1], gene.shape[1]

    features = np.zeros((len(xx), n+m), dtype='float32')
    for i in range(len(xx)):
        features[i,:] = np.concatenate( [cell_type[xx[i,0],:], gene[xx[i,1],:]] ) 

    data = np.concatenate([xx, features], axis=1)
    np.save(os.path.join(folder, f'xx+features_{n}+{m}.npy'), data) 


if __name__ == "__main__":
    folder = os.path.join(PATH_DATA, 'data_pivot_1')
    os.makedirs(folder, exist_ok=True)
    
    create_targets(folder)
    #create_features(folder)