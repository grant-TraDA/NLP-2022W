import pandas as pd
import numpy as np


def prepare_brands_list(train_df, brands_to_drop):
    brands_ = train_df["brand_left"].unique().tolist()
    brands_.extend(train_df["brand_right"].unique().tolist())

    brands_ = [i for i in brands_ if i is not None]
    brands = []
    for brand in brands_:
        brs = brand.split()
        brs = [x.replace('"', '').replace("'", "") for x in brs]
        brands.extend(brs)

    brands = list(set(brands))
    brands = [el for el in brands if el not in brands_to_drop]
    return brands

def brands_in_title_check(dataset, brands):
    train_df_left = dataset[["id_left", "title_left"]]
    train_df_right = dataset[["id_right", "title_right"]]
    train_df_left = train_df_left.drop_duplicates().rename({"id_left" : "id", "title_left" : "title"}, axis = 'columns')
    train_df_right = train_df_right.drop_duplicates().rename({"id_right" : "id", "title_right" : "title"}, axis = 'columns')
    df_train_all = pd.concat([train_df_right, train_df_left])
    df_train_titles = df_train_all.groupby("id").first().reset_index()

    df_train_titles["brand_in_title"] = df_train_titles["title"].apply(lambda x : any(ele in x for ele in brands))

    return df_train_titles

def drop_brands(title, brands):
    for brand in brands:
        title  = title.replace(brand, '')
    return title

def prepare_new_dataset(train_df, brands):
    ids = []
    ids.extend(train_df["id_left"].unique().tolist())
    ids.extend(train_df["id_right"].unique().tolist())
    ids = np.array(list(set(ids)))
    np.random.seed(42)
    remove_brand_mask = np.random.choice([True, False], size =len(ids))
    
    ids_removed_brands = ids[remove_brand_mask]   

    train_df1 = train_df.copy()


    id_remove_left =  train_df1["id_left"].isin(ids_removed_brands).values

    train_df1.loc[id_remove_left, "title_left"] = train_df1.loc[id_remove_left, :].apply(lambda x: drop_brands(x.title_left, brands), axis=1)

    id_remove_right =  train_df1["id_right"].isin(ids_removed_brands).values

    train_df1.loc[id_remove_right, "title_right"] = train_df1.loc[id_remove_right, :].apply(lambda x: drop_brands(x.title_right, brands), axis=1)

    train_df1["changed"] = False
    train_df1["changed"] = train_df1["id_left"].isin(ids_removed_brands)
    train_df1["changed"] = train_df1.apply(lambda x: True if x["id_right"] in (ids_removed_brands) else x["changed"], axis=1)    

    
    return train_df1, ids_removed_brands