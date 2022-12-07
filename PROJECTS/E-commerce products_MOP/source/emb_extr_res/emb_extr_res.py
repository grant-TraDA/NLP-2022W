import pandas as pd

## Functions enabling loading results of embedding extractions

def get_embeddings_df(path):
    df = pd.read_csv(path, sep=';')
    temp = df['embedding'].str.split(',', expand=True)
    temp = temp.apply(pd.to_numeric, axis=1)
    df = df.drop(columns=['embedding'])
    res = pd.concat([df, temp], axis=1, join="inner")
    res = res.groupby(by='id').first().reset_index()
    return res
    
def get_pairs_similarity_df(path):
    df = pd.read_csv(path, sep=',')
    temp = df['pair_id'].str.split('_', expand=True)
    temp.columns = ['right_id', 'left_id']
    df = df.drop(columns=['pair_id'])
    res = pd.concat([df, temp], axis=1, join="inner")
    return res

def get_pretrain_agg_similarity(path):
    return pd.read_csv(path, sep=',')