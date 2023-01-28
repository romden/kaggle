import numpy as np
import pandas as pd
import os
import dill

from config import PATH_INPUT, PATH_WORKING
from dataload import create_data_t7
from preprocessor import build_preprocessor_nn


def load_embeddings(col):
    # col in {cfips, state}
    encoder = dill.load(open(f'{PATH_INPUT}/embeddings/encoder.dill', 'rb'))
    df = pd.read_csv(f'{PATH_INPUT}/aux/states.csv', usecols=[col])
    df = df.drop_duplicates()
    df['label'] = df[col].map(encoder.labels[col])
    embed = np.load(f'{PATH_INPUT}/embeddings/{col}.npy')
    df_embed = pd.DataFrame(embed, columns=[f'{col}_embed_1', f'{col}_embed_2'])
    df_embed['label'] = np.arange(len(embed))
    df = df.merge(df_embed, on='label', how='left').drop(columns=['label'])
    return df


def prepare_data_nn(folder, census=False):    

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    dtrain, dtest = create_data_t7(census)
    
    preprocessor = build_preprocessor_nn(census)
    preprocessor.fit(pd.concat([dtrain, dtest], axis=0))    
    
    y_train = dtrain[[col for col in dtrain.columns if col.startswith('step_')]].values
    y_train = np.concatenate([np.nan_to_num(y_train), 1-np.isnan(y_train)], axis=1) # treat missing values in targets

    x_train = preprocessor.transform(dtrain)
    x_test = preprocessor.transform(dtest)

    np.save(os.path.join(folder, 'y_train.npy'), y_train)
    np.save(os.path.join(folder, 'x_train.npy'), x_train)    
    np.save(os.path.join(folder, 'x_test.npy'), x_test)

    dill.dump(preprocessor, open(os.path.join(folder, 'preprocessor.dill'), 'wb'))


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other-dev/kaggle-03-microbusiness-density-forecasting/dataprep.py
    """
    prepare_data_nn(folder=f'{PATH_WORKING}/n_targets_7_lags_census/data', census=True)