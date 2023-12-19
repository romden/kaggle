import numpy as np
import pandas as pd

import os
import dill
import time
import itertools

from config import *


# path to predictions
paths = {
    'cite': os.path.join(PATH_WORKING, 'selfsv_cite_x512_y140', 'train_ffn_vae', 'y_pred.npy'),
    'multi': os.path.join(PATH_WORKING, 'selfsv_multi_x512_y512', 'train_ffn_vae', 'y_pred.npy')
}


def main():

    def helper(cell_id, gene_id):
        """Generate all combinations of cell_id & gene_id"""
        data = list(itertools.product(cell_id, gene_id))
        return np.array(data, dtype=[('cell_id', np.dtype('O')), ('gene_id', np.dtype('O'))])

    test_ids = dill.load(open(os.path.join(PATH_DATA, 'test_ids.dill'), 'rb'))

    cite_ids = helper(test_ids['cite']['cell_id'], test_ids['cite']['gene_id'])    
    multi_ids = helper(test_ids['multi']['cell_id'], test_ids['multi']['gene_id'])
    predictions = np.concatenate([np.load(paths['cite']).ravel(), np.load(paths['multi']).ravel()])  

    assert len(cite_ids)+len(multi_ids) == len(predictions)

    df = pd.concat([pd.DataFrame(cite_ids), pd.DataFrame(multi_ids)], ignore_index=True)
    df['target'] = predictions

    eval_ids = pd.read_csv(FP_EVALUATION_IDS, dtype={'row_id': int, 'cell_id': str, 'gene_id': str})
    df = eval_ids.merge(df, on=['cell_id', 'gene_id'], how='left')
    df = df[['row_id', 'target']]

    submission = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'), usecols=['row_id'], dtype={'row_id': int})
    submission = submission.merge(df, on='row_id', how='left')
    assert all(submission['target'].notna())
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