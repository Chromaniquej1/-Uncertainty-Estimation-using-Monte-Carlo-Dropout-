import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Drop irrelevant columns
    columns_to_drop = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                       "total_bedrooms", "population", "households"]
    df_train = df_train.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)

    return df_train, df_test

def preprocess_data(df_train, df_test):
    X_train = df_train.drop(["median_house_value"], axis=1).values
    y_train = df_train["median_house_value"].values
    X_test = df_test.drop(["median_house_value"], axis=1).values
    y_test = df_test["median_house_value"].values

    # Normalize the data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, y_train, X_test_norm, y_test
