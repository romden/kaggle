import tensorflow as tf
import numpy as np


def build_model(T, n_feat, model='transformer', params=None):
    models = {
        'transformer': build_transformer(T, n_feat, params),
        'reccurent': build_reccurent(T, n_feat, params),
        }
    return models[model]


class TransformerBlockMod(tf.keras.layers.Layer):
    """Default maybe better: see solution"""
    
    def __init__(self, n_feat, n_heads, n_key, n_ffn, rate=0.1, l2=1e-3):
        super().__init__()
        reg = tf.keras.regularizers.L2(l2)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=n_key, dropout=rate, kernel_regularizer=reg)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_ffn, activation='relu', kernel_regularizer=reg),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(n_feat, kernel_regularizer=reg),
            ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs_lst, training):
        inputs, mask = inputs_lst
        # step 1
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # step 2
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def ff_block(n_units, rate, l2, n_out, act_out):
    return tf.keras.Sequential([
        #tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        #tf.keras.layers.Dropout(rate),
        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.Dropout(rate),
        tf.keras.layers.Dense(n_out, activation=act_out, kernel_regularizer=tf.keras.regularizers.L2(l2))])
        

def get_position_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return tf.cast(P[np.newaxis, ...], dtype=tf.float32)


def treat_missing(x, start_x, end_x, start_m, end_m, n=175):
    # learn embeddings
    emb_inp = tf.tile([[0]], tf.shape(x)[:2])
    emb_out = tf.keras.layers.Embedding(input_dim=1, output_dim=n)(emb_inp)    
    #emb_out = ff_block(n_units=64, rate=0.1, l2=1e-3, n_out=end_x-start_x, act_out='linear')(x[:,:,:end_x])
    # add embeddings using mask
    return tf.concat([x[:,:,:start_x], x[:,:,start_x:end_x] + x[:,:,start_m:end_m] * emb_out], axis=-1)


def causal_call(layer, x, mask, T=14):
    
    B = tf.shape(x)[0]
    outs = []    
    
    for t in range(T):
        if t == 0:
            current_mask = mask
        else:
            lst = [float(i>=t) for i in range(T)]
            m = tf.tile([lst], [B, 1])            
            current_mask = mask *  m[:,:,None] * m[:,None,:]  
            
        out = layer([x, current_mask])            
        outs.append(out[:,t:t+1,:])
        
    return tf.concat(outs, axis=1)


def cosine_similarity_tf(a, b):
    # x shape is n_a * dim
    # y shape is n_b * dim
    # results shape is n_a * n_b
    normalize_a = tf.nn.l2_normalize(a,1)        
    normalize_b = tf.nn.l2_normalize(b,1)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance


def cosine_similarity_score(x):
    batch_size = tf.shape(x)[0]
    idx = tf.range(0, batch_size)
    idx1 = tf.roll(idx, -1, axis=-1)
    idx2 = tf.roll(idx, -2, axis=-1)

    similarity = cosine_similarity_tf(x, x)
    score = tf.gather(similarity, indices=idx1, batch_dims=1) - tf.gather(similarity, indices=idx2, batch_dims=1)
    score = tf.keras.activations.sigmoid(score)
    return score


def contrastive_act(h, h1, tau=0.7):
    dist = tf.matmul(h, h1, transpose_b=True) / tau
    dist_scaled = tf.keras.activations.softmax(dist)
    dist_scaled_diag = tf.linalg.diag_part(dist_scaled)
    return tf.reshape(dist_scaled_diag, [-1,1])


def cutmix_fn(t, pmix=0.7):
    return tf.where(tf.random.uniform(tf.shape(t)) < pmix, t, tf.roll(t, -1, axis=0))
    

def build_transformer(T, n_x, params=None):
    """Transformer with semi supervised pre training"""
    
    x = tf.keras.layers.Input(shape=(T, n_x)) # shape = (batch_size, T, n_x)
    mask = tf.keras.layers.Input(shape=(T, )) # shape = (batch_size, T) mask for present sequence elements

    batch_size = tf.shape(x)[0]
    n_cat = 62 # 59 preprocessor.fit: on train = 59, on train+test=62
    n_num = 175
    n_feat = n_cat + n_num

    assert n_x == n_feat + n_num

    # create attention mask    
    #att_mask = tf.ones([tf.shape(x)[0], T, T]) * mask[:,:,None] * mask[:,None,:] # without CLS
    m = tf.concat([tf.ones([batch_size, 1]), mask], axis=1)
    att_mask = tf.ones([batch_size, T+1, T+1]) * m[:,:,None] * m[:,None,:] # with CLS

    # treat missing values 
    feats = treat_missing(x, start_x=n_cat, end_x=n_feat, start_m=n_feat, end_m=n_feat+n_num)
        
    # add positional embeddings
    #feats = feats + get_position_encoding(seq_len=T, d=n_feat) 

    """add positional embeddings and CLS"""
    rng = tf.range(start=0, limit=T+1, delta=1) 
    pos = tf.tile(tf.reshape(rng, [1, -1]), [batch_size, 1])
    pos_emb = tf.keras.layers.Embedding(input_dim=T+1, output_dim=n_feat)(pos)
    
    feats = tf.concat([pos_emb[:,:1,:], pos_emb[:,1:,:] + feats], axis=1)

    # params
    n_blocks = 2 if params is None else params['n_blocks']
    n_heads = 2 if params is None else params['n_heads']
    n_units = 64 if params is None else params['n_units']
    n_ffn = 64 if params is None else params['n_ffn']
    rate = 0.1 if params is None else params['rate']
    l2 = 0.001 if params is None else params['l2'] # 0.0015, meaningful [1e-3, 1e-2], 1e-2 - restricts to much
    n_causal = 0 #n_blocks - 1     

    trans_blocks = [TransformerBlockMod(n_feat, n_heads, n_units, n_ffn, rate, l2) for i in range(n_blocks)]
    ff_bce = ff_block(n_units=n_units, rate=rate, l2=l2, n_out=1, act_out='sigmoid')
    ff_sim = ff_block(n_units=n_units, rate=rate, l2=l2, n_out=n_units, act_out='linear')
    getcontext = tf.keras.layers.Lambda(lambda t: t[:,0,:])

    def apply_trans(h):
        for i, trans in enumerate(trans_blocks):
            if i < n_causal:
                h = causal_call(trans, h, att_mask)
            else:
                h = trans([h, att_mask])
        h = getcontext(h)
        return h    

    h = apply_trans(feats)
    y = ff_bce(h)

    model = tf.keras.models.Model(inputs=[x, mask], outputs=y)

    # =============================================================
    if True:
        # based on cosine_similarity_score
        o1 = ff_sim(h)
        y1 = tf.reshape(cosine_similarity_score(o1), [-1,1])
    else:
        # based on contrastive loss from SAINT paper
        cutmix = tf.keras.layers.Lambda(cutmix_fn, name='CutMix')
        feats1 = tf.concat([feats[:,:1,:], cutmix(feats[:,1:,:])], axis=1)
        h1 = apply_trans(feats1)
        y1 = contrastive_act(ff_sim(h), ff_sim(h1))
    
    premodel = tf.keras.models.Model(inputs=[x, mask], outputs=tf.concat([y, y1], axis=1))

    return model, premodel


def build_reccurent(T, n_x, params=None):
    """Reccurent with semi supervised based on contrastive loss"""

    n_blocks = 2 if params is None else params['n_blocks']
    n_units = 64 if params is None else params['n_units']
    rate = 0.1 if params is None else params['rate']
    l2 = 0.001 if params is None else params['l2']
    
    x = tf.keras.layers.Input(shape=(T, n_x)) # shape = (batch_size, T, n_x)
    mask = tf.keras.layers.Input(shape=(T, )) # shape = (batch_size, T) mask for present sequence elements

    batch_size = tf.shape(x)[0]
    n_cat = 62 # 59 preprocessor.fit: on train = 59, on train+test=62
    n_num = 175
    n_feat = n_cat + n_num

    assert n_x == n_feat + n_num

    # treat missing values 
    feats = treat_missing(x, start_x=n_cat, end_x=n_feat, start_m=n_feat, end_m=n_feat+n_num)   

    gru = tf.keras.Sequential([
            tf.keras.layers.GRU(n_units, return_sequences=True, dropout=rate, kernel_regularizer=tf.keras.regularizers.L2(l2)),
            tf.keras.layers.GRU(n_units, return_sequences=True, dropout=rate, kernel_regularizer=tf.keras.regularizers.L2(l2)),
            tf.keras.layers.Lambda(lambda t: t[:,-1,:])
        ])
    ff_bce = ff_block(n_units=n_units, rate=rate, l2=l2, n_out=1, act_out='sigmoid')
    ff_sim = ff_block(n_units=n_units, rate=rate, l2=l2, n_out=n_units, act_out='linear')

    h = gru(feats)
    y = ff_bce(h)

    model = tf.keras.models.Model(inputs=[x, mask], outputs=y)

    # =============================================================
    if False:
        # based on cosine_similarity_score
        o1 = ff_sim(h)
        y1 = tf.reshape(cosine_similarity_score(o1), [-1,1])
    else:
        # based on contrastive loss from SAINT paper
        cutmix = tf.keras.layers.Lambda(cutmix_fn, name='CutMix')
        feats1 = cutmix(feats)
        h1 = gru(feats1)
        y1 = contrastive_act(ff_sim(h), ff_sim(h1))
    
    premodel = tf.keras.models.Model(inputs=[x, mask], outputs=tf.concat([y, y1], axis=1))

    return model, premodel


def build_reccurent_base(T, n_x, params=None):
    """Base reccurent model."""

    n_blocks = 2 if params is None else params['n_blocks']
    n_units = 64 if params is None else params['n_units']
    rate = 0.1 if params is None else params['rate']
    l2 = 0.001 if params is None else params['l2']
    
    x = tf.keras.layers.Input(shape=(T, n_x)) # shape = (batch_size, T, n_x)
    mask = tf.keras.layers.Input(shape=(T, )) # shape = (batch_size, T) mask for present sequence elements

    n_cat = 62 # 59 preprocessor.fit: on train = 59, on train+test=62
    n_num = 175
    n_feat = n_cat + n_num

    assert n_x == n_feat + n_num

    # treat missing values 
    feats = treat_missing(x, start_x=n_cat, end_x=n_feat, start_m=n_feat, end_m=n_feat+n_num)

    seq = feats
    for _ in range(n_blocks):
        seq, state = tf.keras.layers.GRU(n_units, return_sequences=True, return_state=True, 
                                         dropout=rate, kernel_regularizer=tf.keras.regularizers.L2(l2))(seq)

    y = ff_block(n_units=n_units, rate=rate, l2=l2, n_out=1, act_out='sigmoid')(state)

    model = tf.keras.models.Model(inputs=[x, mask], outputs=y)

    return model