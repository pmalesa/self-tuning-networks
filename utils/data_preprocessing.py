import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import yaml
import re

'''
    Function that loads the given data and
    returns its feature data and labels separately. 
'''
def load_data(dataset_key: str, config_path: str = "config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        dataset_config = config["datasets"][dataset_key]
        
        data_path = dataset_config["path"]
        label_column = dataset_config.get("label")
        delimiter = dataset_config.get("delimiter")

        if delimiter is None:
            delimiter = ","
        
        df = pd.read_csv(data_path, delimiter = delimiter)

        if label_column is None:
            label_column = df.columns[-1]
        else:
            label_column = label_column.strip()

        X = df.drop(columns = [label_column])
        y = df[label_column]

        return X, y

'''
    Function that converts a date column to elapsed
    time from the earliest date in the column.
'''
def convert_date_to_elapsed(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df[date_column] = pd.to_datetime(df[date_column])
    reference_date = df[date_column].min()
    df[date_column] = (df[date_column] - reference_date).dt.total_seconds()
    return df

'''
    Function that converts any categorical feature
    columns to one-hot numerical columns.
'''
def convert_to_numerical(df: pd.DataFrame) -> np.ndarray:
    # Convert columns that are expected to be datetime but are of type object
    for col in df.columns:
        if df[col].dtype == "object" and is_date_format(df[col]):
            df[col] = pd.to_datetime(df[col])

    # Find date columns
    date_cols = df.select_dtypes(include = ["datetime64", "datetime64[ns]"]).columns.tolist()

    # Convert date columns to elapsed time
    for col in date_cols:
        df = convert_date_to_elapsed(df, col)

    # Find categorical columns
    categorical_cols = df.select_dtypes(include = ["object", "category"]).columns.tolist()

    # Apply one-hot encoding to categorical columns
    preprocessor = ColumnTransformer(
        transformers = [
            ("onehot", OneHotEncoder(), categorical_cols)
        ],
        remainder = "passthrough"
    )

    # Transform the data
    transformed_data = preprocessor.fit_transform(df)

    # Convert the resulting data to a numpy array
    if not isinstance(transformed_data, np.ndarray):
        transformed_data = transformed_data.toarray()

    return transformed_data

'''
    Function that normalizes the feature
    data using the chosen scaler.
'''
def normalize_numerical_data(X: np.ndarray, scaler = "minmax") -> np.ndarray:
    if scaler == "standard":
        scaler = StandardScaler()
        return  scaler.fit_transform(X)
    elif scaler == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    else:
        print(f"Error, no such scaler as '{scaler}'! Use 'standard' or 'minmax'.")
        return None

'''
    Function that converts any categorical feature data to
    numerical and normalizes it using chosen scaler (minmax or standard).
'''
def preprocess_feature_data(df: pd.DataFrame, scaler = "minmax") -> np.ndarray:
    numerical_data = convert_to_numerical(df)

    # Add imputation step to handle NaN values using median strategy
    imputer = SimpleImputer(strategy = "median")
    numerical_data_imputed = imputer.fit_transform(numerical_data)

    return normalize_numerical_data(numerical_data_imputed, scaler)

'''
    Function that converts labels, either categorical or not
    to integers starting from 0.
'''
def preprocess_label_data(df: pd.DataFrame) -> np.ndarray:
    label_encoder = LabelEncoder()
    y_transformed = label_encoder.fit_transform(df.values.ravel())
    label_mapping = {new: original for original, new in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    return y_transformed, label_mapping

'''
    Function that utilizes the functions implemented above and
    returns the preprocessed feature data, label data and label mapping.
    It returns both preprocessed feature and label data with mappings in
    case of classification problem and preprocessed feature data with
    original label date in case of regression problem.
'''
def preprocess_data(X: pd.DataFrame, y: pd.DataFrame, regression: bool = False):
    X_preprocessed = preprocess_feature_data(X)
    if regression:
        # Normalization of target values to prevent too big loss values (NaN)
        scaler_y = StandardScaler()
        y_preprocessed = scaler_y.fit_transform(y.to_numpy().reshape(-1, 1)).flatten()
        return X_preprocessed, y_preprocessed
    else:
        y_preprocessed, label_mapping = preprocess_label_data(y)
        return X_preprocessed, y_preprocessed, label_mapping

'''
    Function that checks if the pandas series is of "yyyy-mm-dd" format.
'''
def is_date_format(s: pd.Series) -> bool:
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return s.dropna().apply(lambda x: bool(date_pattern.match(x))).all()

'''
  Function that splits data into training, validation, and test data sets and
  returns the result of splitting.
'''
def train_val_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.15, train_size: float = 0.70, val_size: float = 0.15, validate: bool = False, random_state = None):
    if not validate:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
        return X_train, X_test, y_train, y_test
    elif validate and train_size + val_size + test_size == 1.0:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = (1 - train_size), random_state = random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = (test_size / (test_size + val_size)), random_state = random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        raise ValueError(f'Split proportions {test_size}, {val_size}, {train_size} do not add up.')
