import pandas as pd
from sklearn.model_selection import train_test_split
from real_estate_price import config

def load_and_prepare_data():
    """
    Loads real estate data, splits into X/y and train/test sets.
    """
    data = pd.read_csv(config.DATA_PATH)

  
    X = data.drop(['price','property_age'], axis=1)
    y = data['price']

  
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("Data loaded and split successfully.")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
