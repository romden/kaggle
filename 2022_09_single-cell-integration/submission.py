import numpy as np
import pandas as pd

import os
import dill

from config import *

# true training targets
trues = {
    'cite': pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32'),
    'multi': pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32')
}

# test predictions
paths = {
    #'cite': os.path.join(PATH_WORKING, 'data_cite_x512', 'y_pred.npy'), 
    #'cite': os.path.join(PATH_WORKING, 'nnlgb_cite_x512', 'y_pred.npy'),
    'cite': os.path.join(PATH_WORKING, 'pdl_cite_x512', 'y_pred.npy'),
    'multi': os.path.join(PATH_WORKING, 'nn_multi_x1024', 'y_pred.npy')
}


def predictions_df(cell_ids, gene_ids, values):    
    data = {'cell_id': [], 'gene_id': [], 'target': []}    
    for i, cell_id in enumerate(cell_ids):
        data['cell_id'] += [cell_id]*len(gene_ids)
        data['gene_id'] += gene_ids
        data['target'] += values[i,:].tolist()    
    return pd.DataFrame(data)


def main():    
    
    data = dill.load(open(os.path.join(PATH_DATA, 'test_ids.dill'), 'rb'))
    
    dfs = []    
    for key in ['cite', 'multi']:
        # load predictions
        y_pred = np.load(paths[key])
        # clip predictions to the range seen in train data
        if False: 
            lb = np.min(trues[key], axis=0, keepdims=True)
            ub = np.max(trues[key], axis=0, keepdims=True)
            y_pred = np.clip(y_pred, lb, ub)
        # make pandas dataframe
        df = predictions_df(cell_ids=data[key]['cell_id'], gene_ids=data[key]['gene_id'], values=y_pred)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    eval_ids = pd.read_csv(FP_EVALUATION_IDS, dtype={'row_id': int, 'cell_id': str, 'gene_id': str})
    df = eval_ids.merge(df, on=['cell_id', 'gene_id'], how='left')
    df = df[['row_id', 'target']]

    submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), usecols=['row_id'], dtype={'row_id': int})
    submission = submission.merge(df, on='row_id', how='left')
    assert all(submission['target'].notna()) and len(submission.columns) == 2
    submission['target'] = submission['target'].round(6) # reduce the size of the csv    
    submission.to_csv(os.path.join(PATH_WORKING, 'sample_submission.csv'), index=False)  
    print('submission created.')      


if __name__ == "__main__":
    """ 
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/submission.py
    """
    main()


# python -c "import numpy as np; print(np.__version__)"
# numpy version in cpu and gpu: 1.19.5  1.20.3