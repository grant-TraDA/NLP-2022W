import json
import ast

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix


def encode_results():
    results_data = pd.read_csv(
        "results/specified_dietary_tags_classification_v3.csv"
    )
    results_data_transformed_rows = []
    for _, row in results_data.iterrows():
        ingrds = row["gpt-3_classification"]
        # Postprocessing results to only extract json with model Classification
        ingrds = ingrds[ingrds.find("{") :]
        json_row = json.loads(ingrds)
        results_data_transformed_rows.append(json_row)

    return pd.DataFrame(results_data_transformed_rows).astype(int)


def save_classification_results(results_df, target_data):
    for tag in results_df.columns:
        # Calculating classification reports
        r = classification_report(
            y_true=target_data[f"{tag}_gt"], y_pred=results_df[tag]
        )
        with open(
            f"results/classification_results/{tag}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(r)
        # Calculating confusion matrices
        conf_mat = confusion_matrix(
            y_true=target_data[f"{tag}_gt"], y_pred=results_df[tag]
        )
        df_cfm = pd.DataFrame(conf_mat)
        plt.figure(figsize=(10, 10))
        cfm_plot = sns.heatmap(df_cfm, annot=True)
        cfm_plot.figure.savefig(f"results/classification_results/{tag}.png")


def evaluate_classification_results():
    results_df = encode_results()
    target_data = (
        pd.read_csv("results/llm_sampled_data.csv").iloc[:, :-2].astype(int)
    )
    target_data.columns = [f"{col}_gt" for col in target_data.columns]

    target_data.head()

    target_data = target_data.iloc[: results_df.shape[0]]

    save_classification_results(results_df, target_data)


def merge_ingredients(gt_data, predicted_data):
    ret_df = pd.DataFrame()
    ret_df["gt"] = gt_data.ingredients_list
    ret_df["pred"] = predicted_data["gpt-3_extracted"]

    return ret_df


def calculate_metric_ingredients(ingredients_results):
    perc_pred_in_gt_all = []
    perc_gt_in_pred_all = []
    iou_all = []
    for _, row in ingredients_results.iterrows():
        gt = row["gt"]
        pred = row["pred"]
        len_gt = len(gt)
        len_pred = len(pred)
        # Calculating if prediction elements are in ground truth
        perc_pred_in_gt = 0
        for el in pred:
            if el in gt:
                perc_pred_in_gt += 1
        perc_pred_in_gt /= len_gt
        perc_pred_in_gt_all.append(perc_pred_in_gt)
        # Calculating if ground truth is in prediction
        perc_gt_in_pred = 0
        for el in gt:
            if el in pred:
                perc_gt_in_pred += 1
        perc_gt_in_pred /= len_pred
        perc_gt_in_pred_all.append(perc_gt_in_pred)
        # Calculating IoU
        iou_all.append(
            len(set(gt).intersection(set(pred)))
            / len(set(gt).union(set(pred)))
        )

    return pd.DataFrame(
        {
            "%_gt_in_pred": perc_gt_in_pred_all,
            "%_pred_in_gt": perc_pred_in_gt_all,
            "iou": iou_all,
        }
    )


def evaluate_extraction_results():
    # Loading data
    gt_data = pd.read_csv("llm_sampled_data.csv")
    predicted_data = pd.read_csv("ingredients_extraction_results.csv")
    ingredients_results = merge_ingredients(gt_data, predicted_data)

    ingredients_results.dropna(inplace=True)
    # Casting strings into sets for evaluation
    ingredients_results["gt"] = ingredients_results["gt"].str.split(", ")

    ingredients_results["pred"] = ingredients_results["pred"].apply(
        ast.literal_eval
    )

    ingredients_results.head()

    metrics_ingredients = calculate_metric_ingredients(ingredients_results)

    metrics_ingredients.to_csv(
        "results/ingredients_results/ingredients_results.csv"
    )

    ingredients_ret_df = (
        metrics_ingredients.mean(axis=0)
        .reset_index()
        .rename(columns={0: "avg", "index": "metric"})
    )

    ingredients_ret_df.to_csv(
        "results/ingredients_results/ingredients_result_metrics.csv"
    )


if __name__ == "__main__":
    evaluate_classification_results()
    evaluate_extraction_results()
