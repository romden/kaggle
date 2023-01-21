import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_generator(X, y, batch_size, goal):
    if goal == 'train':
        generator = simple_generator(X, y, batch_size=batch_size, goal=goal) #, oversampling={0: 10, 1: 1}
    elif goal == 'pretrain':
        generator = simple_generator(X, y, batch_size=batch_size, goal=goal)       
    elif goal == 'predict':
        generator = simple_generator(X, y, batch_size=batch_size, shuffle=False, goal=goal)
    return generator


def create_batch(X, y, batch_indexes, goal=None):

    feats, mask = [x[batch_indexes] for x in X]
     
    x_batch = (feats, mask)
    y_batch = y[batch_indexes]

    if goal == 'pretrain' and True:

        batch_size = feats.shape[0]     
        values = feats.reshape(batch_size, -1)
        similarity = cosine_similarity(values) # in [0,1], 1 - is the most similar
        
        idx = np.arange(batch_size)
        idx1 = np.roll(idx, -1)
        idx2 = np.roll(idx, -2)
        y_sim = (similarity[idx, idx1] > similarity[idx, idx2]).astype('float32').reshape(-1,1)

        y_batch = np.concatenate([y_batch, y_sim], axis=1)

    if False: # use for RNN
        m = mask.shape[1]
        idx = np.flip(np.arange(m))
        x_batch = (feats[:,idx,:], mask[:,idx])

    return (x_batch, y_batch)


def simple_generator(X, y, batch_size=64, shuffle=True, drop_remainder=True, oversampling=None, seed=1000, goal='train'):
    """The outer enclosing function."""

    np.random.seed(seed)
    
    def wrapper():
        """The nested function that generates the data."""

        # create an array with indexes
        if oversampling:
            # oversampling [dict] = {0: n0, 1: n1}
            n0, n1 = oversampling[0], oversampling[1]
            idx0, idx1 = np.where(y.ravel() < 1)[0], np.where(y.ravel() > 0)[0]        
            indexes = np.concatenate([np.repeat(idx0, n0), np.repeat(idx1, n1)])
        else:
            indexes = np.arange(len(y))
        
        if shuffle:
            np.random.shuffle(indexes)

        # init params        
        batch_indexes = [0]*batch_size
        data_size = len(indexes)
        idx = 0 # current location
        flag = True # indicates if there is next batch

        while flag:

            for i in range(batch_size):
                if idx >= data_size:
                    flag = False
                    idx = 0
                    if shuffle:
                        np.random.shuffle(indexes)                
                batch_indexes[i] = indexes[idx]            
                idx += 1

                # drop remainder if needed
                if idx == data_size and drop_remainder:
                    batch_indexes = batch_indexes[:i+1]
                    flag = False
                    break
            
            batch = create_batch(X, y, batch_indexes, goal)

            yield batch
        
    return wrapper


def sampler_generator(X, y, n0, n1, seed=1000):
    """The outer enclosing function. 
        Generates batchs of data based on the proportion determined by n0 and n1.
        [Note]: Do not use for all cases! works for appropriate distributions of n0 and n1.
    
    Args:
        X: features
        y: labels
        n0: number of points with label 0 in the batch
        n1: number of points with label 1 in the batch
    Returns:
        wrapper: generator
    """

    def sample_indexes(idx, batch_size, batch_indexes, indexes):
        n = len(indexes)
        flag = True
        for i in range(batch_size):
            if idx >= n:
                flag = False
                idx = 0
                np.random.shuffle(indexes)                              
            batch_indexes[i] = indexes[idx]            
            idx += 1  
        return idx, flag   

    # create an array with indexes
    indexes0 = np.where(y.ravel() < 1)[0]
    indexes1 = np.where(y.ravel() > 0)[0]

    # init 
    batch_indexes0 = [0]*n0
    batch_indexes1 = [0]*n1
    mid = n1 // 2

    np.random.seed(seed)

    def wrapper():
        """The nested function that generates the data."""
        # shuffle 
        np.random.shuffle(indexes0)
        np.random.shuffle(indexes1)
        
        # current location 
        i0 = 0
        i1 = 0 
        flag = True

        while flag:

            i0, flag = sample_indexes(idx=i0, batch_size=n0, batch_indexes=batch_indexes0, indexes=indexes0)
            i1, _    = sample_indexes(idx=i1, batch_size=n1, batch_indexes=batch_indexes1, indexes=indexes1)

            batch_indexes = batch_indexes1[:mid] + batch_indexes0 + batch_indexes1[mid:]

            batch = create_batch(X, y, batch_indexes, pretrain=False)

            yield batch
        
    return wrapper