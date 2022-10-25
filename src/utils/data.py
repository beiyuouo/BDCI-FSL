import os
import json
import pandas as pd


def load_json(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df


def load_csv(file_path: str):
    df = pd.read_csv(file_path, encoding="utf-8", index_col=0)
    return df


def load_train_df(file_path: str):
    if file_path.endswith(".json"):
        df = load_json(file_path)
    elif file_path.endswith(".csv"):
        df = load_csv(file_path)
    else:
        raise ValueError("file path must be json or csv")

    df["label_id"] = df["label_id"].apply(lambda x: int(x))
    df.reset_index(inplace=True)
    return df


def load_test_df(file_path: str):
    df = load_json(file_path)
    df.reset_index(inplace=True)
    return df


def load_data(data_path: str, split: str = "train"):
    if split == "train":
        train_df = load_train_df(os.path.join(data_path, "train_en.csv"))
        return train_df
    elif split == "test":
        test_df = load_test_df(os.path.join(data_path, "testA_en.json"))
        return test_df
    else:
        train_df = load_train_df(os.path.join(data_path, "train_en.csv"))
        test_df = load_test_df(os.path.join(data_path, "testA_en.json"))
        return train_df, test_df


def get_two_stage_data(data_df: pd.DataFrame):
    # is label_id == 2 or not
    stage_one_df = data_df.copy()
    stage_one_df["label_id"] = stage_one_df["label_id"].apply(
        lambda x: 1 if x == 2 else 0
    )

    # other label

    stage_two_df = data_df[data_df["label_id"] != 2].copy()
    stage_two_df["label_id"] = stage_two_df["label_id"].apply(
        lambda x: x if x < 2 else x - 1
    )

    return stage_one_df, stage_two_df
