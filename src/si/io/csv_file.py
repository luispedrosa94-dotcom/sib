import pandas as pd
from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False) -> Dataset:
    header = 0 if features else None
    data = pd.read_csv(filename, sep=sep, header=header)

    if label:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        features_names = data.columns[:-1].tolist() if features else None
        label_name = str(data.columns[-1]) if features else None
    else:
        X = data.values
        y = None
        features_names = data.columns.tolist() if features else None
        label_name = None

    return Dataset(X=X, y=y, features=features_names, label=label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    data = pd.DataFrame(dataset.X)

    if features and dataset.features is not None:
        data.columns = dataset.features

    if label and dataset.has_label():
        label_col_name = dataset.label if (features and dataset.label is not None) else len(data.columns)
        data[label_col_name] = dataset.y

    data.to_csv(filename, sep=sep, index=False, header=features)