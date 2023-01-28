import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from config import PATH_INPUT


def create_lags(df, pivot='state', n_lags=7, offset_months=1):
    
    df = df[[pivot, 'first_day_of_month', 'microbusiness_density']].copy()
    df['first_day_of_month'] = pd.to_datetime(df['first_day_of_month'])
    df = df.set_index([pivot, 'first_day_of_month'])
    tmp = df.copy()   

    for i in range(1, n_lags+1):
        tmp = tmp.reset_index()
        tmp['first_day_of_month'] = tmp['first_day_of_month'] + pd.DateOffset(months=offset_months)
        tmp = tmp.set_index([pivot, 'first_day_of_month'])
        df[f'{pivot}_lag_{i}'] = tmp['microbusiness_density']   
    
    df = df.reset_index().drop(columns=['microbusiness_density'])
    df['first_day_of_month'] = df['first_day_of_month'].astype(str)
    
    return df


def create_data_t7(census=False):
    """Train df with 7 targets."""

    # load train csv
    df_train = pd.read_csv(f'{PATH_INPUT}/train.csv')
    df_test = pd.read_csv(f'{PATH_INPUT}/test.csv')
    df_test = df_test.merge(df_train[['cfips', 'state']].drop_duplicates(), on='cfips', how='left')
    
    df_all = pd.concat([df_train[['cfips', 'state', 'first_day_of_month', 'microbusiness_density']], 
                         df_test[['cfips', 'state', 'first_day_of_month']]], axis=0).reset_index(drop=True)
    
    df_all_state = df_all.groupby(['state', 'first_day_of_month'])['microbusiness_density'].mean().reset_index()
    
    df_state_lags = create_lags(df_all_state, pivot='state', n_lags=7)
    df_cfips_lags = create_lags(df_all, pivot='cfips', n_lags=7)
    df_targets = create_lags(df_all, pivot='cfips', n_lags=6, offset_months=-1)
    df_targets.insert(2, 'lag_0', df_all['microbusiness_density'])  
    df_targets = df_targets.rename(columns={col: f'step_{i}' for i, col in enumerate(df_targets.columns[2:])})
    
    df_all = df_all.drop(columns=['microbusiness_density'])
    df_all.insert(3, 'month', pd.to_datetime(df_all['first_day_of_month']).dt.month)

    # join train csv and profiles
    df = (df_all.merge(df_state_lags, on=['state', 'first_day_of_month'], how='left')
                  .merge(df_cfips_lags, on=['cfips', 'first_day_of_month'], how='left')
                  .merge(df_targets, on=['cfips', 'first_day_of_month'], how='left'))

    if census:
        df = df.merge(pd.read_csv(f'{PATH_INPUT}/census_starter.csv'), on=['cfips'], how='left')
    
    dtrain, dtest = df[:len(df_train)], df[df['first_day_of_month']=='2022-11-01']
    
    # drop missing profile data
    dtrain = dtrain.dropna(how='all', subset=[col for col in df.columns if '_lag_' in col])
    
    dtrain = dtrain.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    dtest = dtest.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)

    return dtrain, dtest