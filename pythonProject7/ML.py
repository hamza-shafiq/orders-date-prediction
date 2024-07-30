import os
import logging
import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(file_name):
    """
    Load CSV data into a DataFrame.

    Args:
        file_name (str): Name of the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Preprocess the DataFrame by handling missing values, converting datatypes, and extracting features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["datestamp"] = pd.to_datetime(df["datestamp"], utc=True)
    df["category"] = df["category"].fillna("Unknown")
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["season"] = df["timestamp"].dt.month % 12 // 3 + 1
    df["total_revenue"] = df["quantity"] * df["price"]
    df["days_since_last_order"] = df.groupby("external_customer_id")[
        "timestamp"
    ].transform(lambda x: x.diff().dt.days.fillna(0))
    return df


def calculate_time_between_orders(df):
    """
    Calculate the time between orders for each customer.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the time between orders calculated.
    """
    df.sort_values(by=["external_customer_id", "timestamp"], inplace=True)
    df["time_between_orders"] = df.groupby("external_customer_id")["timestamp"].diff()
    return df


def aggregate_customer_data(df):
    """
    Aggregate data at the customer level.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Aggregated customer data.
    """
    customer_data = df.groupby("external_customer_id").agg(
        {
            "quantity": "sum",
            "total_revenue": "sum",
            "timestamp": ["min", "max"],
            "time_between_orders": "mean",
        }
    )
    customer_data.columns = [
        "_".join(col).strip() for col in customer_data.columns.values
    ]
    return customer_data


def analyze_ordering_patterns(df):
    """
    Identify patterns in ordering behavior.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: Order frequency by customer and preferred days for orders.
    """
    order_frequency = df["external_customer_id"].value_counts().reset_index()
    order_frequency.columns = ["customer_id", "order_count"]
    preferred_days = df.groupby("day_of_week").size().reset_index(name="order_count")
    return order_frequency, preferred_days


def handle_outliers(df):
    """
    Visualize and remove outliers in the 'price' column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(df["price"])
    plt.title("Price Outliers")
    plt.savefig("plots/price_outliers.png")
    plt.show()

    q_low = df["price"].quantile(0.01)
    q_hi = df["price"].quantile(0.99)
    df_filtered = df[(df["price"] > q_low) & (df["price"] < q_hi)]
    return df_filtered


def preprocess_features(df):
    """
    Preprocess features for model training.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: Processed feature matrix (X) and target vector (y).
    """
    X = df.drop(
        columns=["external_customer_id", "total_revenue", "timestamp", "datestamp"]
    )
    y = df["total_revenue"]

    categorical_features = [
        "order_source",
        "order_status",
        "product_source",
        "category",
        "day_of_week",
    ]
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y


def train_models(x_train, y_train, x_test, y_test):
    """
    Train various models and return their performance metrics.

    Args:
        x_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        x_test (np.ndarray): Testing feature matrix.
        y_test (np.ndarray): Testing target vector.

    Returns:
        dict: Performance metrics for each model.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"MSE": mse, "MAE": mae}

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} Predictions vs Actual Values")
        plt.savefig(f"plots/{name}_predictions_vs_actual.png")
        plt.show()

        logging.info(f"{name} Results: MSE = {mse}, MAE = {mae}")

    return results


def predict_next_order_date(model, df, days_before_notification=3):
    """
    Predict the next order date and calculate notification dates.

    Args:
        model: Trained model for predicting days to next order.
        df (pd.DataFrame): Input DataFrame.
        days_before_notification (int): Number of days before the predicted order date to notify.

    Returns:
        pd.DataFrame: DataFrame with predicted next order dates and notification dates.
    """
    df_copy = df.copy()
    df_copy["days_since_last_order"] = df_copy.groupby("external_customer_id")[
        "timestamp"
    ].transform(lambda x: x.diff().dt.days.fillna(0))
    X_predict, _ = preprocess_features(df_copy)
    df_copy["predicted_days_to_next_order"] = model.predict(X_predict)
    df_copy["predicted_next_order_date"] = df_copy["timestamp"] + pd.to_timedelta(
        df_copy["predicted_days_to_next_order"], unit="D"
    )
    df_copy["notification_date"] = df_copy["predicted_next_order_date"] - pd.Timedelta(
        days=days_before_notification
    )
    return df_copy[
        ["external_customer_id", "predicted_next_order_date", "notification_date"]
    ].drop_duplicates()


def update_predictions_periodically(file_path):
    """
    Update predictions and notification dates at regular intervals.

    Args:
        file_path (str): Path to the CSV file.
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")

    while True:
        try:
            df = load_data(file_path)
            df = preprocess_data(df)
            df = calculate_time_between_orders(df)
            df_filtered = handle_outliers(df)
            X, y = preprocess_features(df_filtered)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)
            df_predictions = predict_next_order_date(model, df_filtered)
            logging.info("Updated Next Order Dates and Notification Dates")
            logging.info(df_predictions.head())

            output_path = "next_order_dates_with_notifications_date.csv"
            df_predictions.to_csv(output_path, index=False)
            logging.info(f"Predictions and notifications saved to {output_path}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")

        time.sleep(86400)


def main():
    """
    Main function to execute the data loading, preprocessing, model training, and predictions.
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")

    file_path = "abc.csv"

    df = load_data(file_path)
    logging.info("Basic Info and Summary Statistics:")
    logging.info(df.info())
    logging.info(df.describe())

    df = preprocess_data(df)
    logging.info("Time-Based Features Added:")
    logging.info(
        df[
            [
                "timestamp",
                "day_of_week",
                "season",
                "total_revenue",
                "days_since_last_order",
            ]
        ].head()
    )

    df = calculate_time_between_orders(df)
    customer_data = aggregate_customer_data(df)
    logging.info("Customer-Level Data:")
    logging.info(customer_data.head())

    order_frequency, preferred_days = analyze_ordering_patterns(df)
    logging.info("Order Frequency by Customer:")
    logging.info(order_frequency.head())
    logging.info("Preferred Days for Orders:")
    logging.info(preferred_days)

    df_filtered = handle_outliers(df)

    X, y = preprocess_features(df_filtered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results = train_models(X_train, y_train, X_test, y_test)
    logging.info("Model Performance Metrics:")
    logging.info(results)

    update_predictions_periodically(file_path)


if __name__ == "__main__":
    main()
