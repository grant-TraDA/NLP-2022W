def get_pair_similarity_probing_task_df(df, embeddings_df):
    pair_ids = df.loc[:, ["id_left", "id_right", "label"]]
    new_df = pair_ids.merge(
        embeddings_df.add_prefix("left_"), left_on=["id_left"], right_on=["left_id"]
    )
    new_df = new_df.merge(
        embeddings_df.add_prefix("right_"), left_on=["id_right"], right_on=["right_id"]
    )
    new_df.drop(columns=["id_right", "id_left"], index=1, inplace=True)
    return new_df
