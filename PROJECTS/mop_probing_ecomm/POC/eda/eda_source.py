import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def _get_pairs(df, type='positive'):
    cond = df['cluster_id_left'] != df['cluster_id_right']
    cond = cond if type == 'positive' else ~(cond)
    return df.loc[cond, :]


def get_positive_pairs(df):
    return _get_pairs(df, type='positive')


def get_negative_pairs(df):
    return _get_pairs(df, type='negative')
    

def get_positive_pairs_count(df):
    return get_positive_pairs(df).shape[0]


def get_negative_pairs_count(df):
    return get_negative_pairs(df).shape[0]


def get_number_of_nans(df, col):
    return df[col].isna().sum() 


def get_number_of_nans_pos_and_neg(df, prefix_col):
    cond = df[f'{prefix_col}_right'].isna() & df[f'{prefix_col}_left'].isna()
    return cond.sum()


def plot_positive_vs_negative(df):
    
    labels = ['Positive', 'Negative']
    values = [get_positive_pairs_count(df), get_negative_pairs_count(df)]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title='Positive and Negative pairs.')
    fig.show()


def plot_missing_values(missing_values, n):
    x = missing_values.keys()
    y = missing_values.values()
    color = ['left', 'right'] * (len(missing_values.items()) // 2)
    x = ['_'.join(v.split('_')[:-1]) for v in list(x)]

    mv = pd.DataFrame({'x': x, 'y': y, 'color': color, 'text': [f"{round(100 * i / n, 2)}% of all" for i in y]})
    fig = px.bar(mv, x='x', y='y', color='color', text='text', barmode='group')
    fig.update_xaxes(title='Feature name')
    fig.update_yaxes(title='Number of missing values')
    fig.show()

def print_missing_values(df, col, missing_values):
    n = df.shape[0]
    x = get_number_of_nans(df, col)
    missing_values.update({col: x})
    print(f"Number of missing values in the column: {col} = {x}, which is {round(x / n * 100, 2)}% of all pairs>")


def plot_missing_values_simultaneously(missing_values, n):
    x = missing_values.keys()
    y = missing_values.values()
    mv = pd.DataFrame({'x': x, 'y': y, 'text': [f"{round(100 * i / n, 2)}% of all" for i in y]})
    fig = px.bar(mv, x='x', y='y', text='text', barmode='group')
    fig.update_xaxes(title='Feature name')
    fig.update_yaxes(title='Number of missing values')
    fig.update_layout(title='Missing values per feature (missing for the right and left offer at the same time)')
    fig.show()

def print_missing_values_simultaneously(df, col_prefix, missing_values):
    n = df.shape[0]
    x = get_number_of_nans_pos_and_neg(df, col_prefix)
    missing_values.update({col_prefix: x})
    print(f"Number of missing values in the column: (both {col_prefix}_right and  {col_prefix}_left): = {x}, which is {round(x / n * 100, 2)}% of all pairs>")

def get_avg_length_feature(df, col, pairs_mode='both'):

    if pairs_mode == 'neg':
        df = get_negative_pairs(df)
    elif pairs_mode == 'pos':
        df = get_positive_pairs(df)
        
    return df[col].fillna(value='').apply(len).mean()


def plot_avg_lengths(avg_dic_all, avg_dic_pos, avg_dic_neg, n):
    x = list(avg_dic_all.keys()) + list(avg_dic_pos.keys()) + list(avg_dic_neg.keys())
    y = list(avg_dic_all.values()) + list(avg_dic_pos.values()) + list(avg_dic_neg.values())
    color = ['all'] * len(list(avg_dic_all.keys())) + ['pos'] * len(list(avg_dic_pos.keys())) + ['neg'] * len(list(avg_dic_neg.keys()))
    text = [f'{round(i, 2)}' for i in y]

    mv = pd.DataFrame({'x': x, 'y': y, 'color': color})
    fig = px.bar(mv, x='x', y='y', color='color', text=text, barmode='group')
    fig.update_xaxes(title='Feature name')
    fig.update_yaxes(title='Avg. length')
    fig.update_layout(title='Avg. length of feature (for all, positive and negative pairs)')
    fig.show()


def print_avg_lengths(df, col, avg_lens, avg_lens_pos, avg_lens_neg):
    x = get_avg_length_feature(df, col=col)
    x_pos = get_avg_length_feature(df, col=col, pairs_mode='pos')
    x_neg = get_avg_length_feature(df, col=col, pairs_mode='neg')

    avg_lens.update({col: x})
    avg_lens_pos.update({col: x_pos})
    avg_lens_neg.update({col: x_neg})

    print(f'The average number of the column: {col} for - all pairs = {x}, pos. pairs = {x_pos}, neg. pairs = {x_neg}')
