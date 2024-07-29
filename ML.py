import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
import time
import os


def load_data(file_path):
    """
    Load CSV data into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
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
    # Convert 'timestamp' from Unix timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # Convert 'datestamp' from object to datetime, handling mixed time zones
    df["datestamp"] = pd.to_datetime(df["datestamp"], utc=True)

    # Fill missing values in 'category' with 'Unknown'
    df["category"] = df["category"].fillna("Unknown")

    # Extract year, month, and day from 'timestamp'
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    # Create additional time-based features
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["season"] = df["timestamp"].dt.month % 12 // 3 + 1  # 1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall

    # Calculate 'total_revenue'
    df["total_revenue"] = df["quantity"] * df["price"]

    # Create feature for days since last order
    df["days_since_last_order"] = df.groupby("external_customer_id")["timestamp"].transform(lambda x: x.diff().dt.days.fillna(0))

    return df


def calculate_time_between_orders(df):
    """
    Calculate the time between orders for each customer.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the time between orders calculated.
    """
    # Sort the data by customer and timestamp
    df.sort_values(by=["external_customer_id", "timestamp"], inplace=True)

    # Calculate the time difference between consecutive orders
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
    # Aggregate data
    customer_data = df.groupby("external_customer_id").agg(
        {
            "quantity": "sum",
            "total_revenue": "sum",
            "timestamp": ["min", "max"],
            "time_between_orders": "mean",
        }
    )

    # Flatten MultiIndex columns
    customer_data.columns = ["_".join(col).strip() for col in customer_data.columns.values]

    return customer_data


def analyze_ordering_patterns(df):
    """
    Identify patterns in ordering behavior.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: Order frequency by customer and preferred days for orders.
    """
    # Ordering frequency
    order_frequency = df["external_customer_id"].value_counts().reset_index()
    order_frequency.columns = ["customer_id", "order_count"]

    # Preferred days
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
    # Visualize outliers using a boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(df["price"])
    plt.title("Price Outliers")
    plt.savefig("plots/price_outliers.png")  # Save the plot
    plt.show()

    # Define quantiles for outlier detection
    q_low = df["price"].quantile(0.01)
    q_hi = df["price"].quantile(0.99)

    # Filter out outliers
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
    # Define features and target
    X = df.drop(columns=["external_customer_id", "total_revenue", "timestamp", "datestamp"])
    y = df["total_revenue"]

    # Define categorical and numerical features
    categorical_features = ["order_source", "order_status", "product_source", "category", "day_of_week"]
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create preprocessing pipeline
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

    # Transform features
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y


def train_models(X_train, y_train, X_test, y_test):
    """
    Train various models and return their performance metrics.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_test (np.ndarray): Testing feature matrix.
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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {"MSE": mse, "MAE": mae}

        # Plot and save model performance
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} Predictions vs Actual Values")
        plt.savefig(f"plots/{name}_predictions_vs_actual.png")
        plt.show()

        print(f"\n{name} Results:")
        print(f"  Mean Squared Error (MSE): {mse}")
        print(f"  Mean Absolute Error (MAE): {mae}")

    return results
#as linear regression has smallest error, so i used it for further predictions.

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
    # Prepare features for prediction
    df_copy = df.copy()  # Work on a copy to avoid modifying the original DataFrame
    df_copy["days_since_last_order"] = df_copy.groupby("external_customer_id")["timestamp"].transform(lambda x: x.diff().dt.days.fillna(0))

    X_predict, _ = preprocess_features(df_copy)

    # Predict days to next order
    df_copy["predicted_days_to_next_order"] = model.predict(X_predict)

    # Calculate the next predicted order date
    df_copy["predicted_next_order_date"] = df_copy["timestamp"] + pd.to_timedelta(df_copy["predicted_days_to_next_order"], unit="D")

    # Calculate notification date
    df_copy["notification_date"] = df_copy["predicted_next_order_date"] - pd.Timedelta(days=days_before_notification)

    return df_copy[["external_customer_id", "predicted_next_order_date", "notification_date"]].drop_duplicates()


def update_predictions_periodically(file_path):
    """
    Update predictions and notification dates at regular intervals.

    Args:
        file_path (str): Path to the CSV file.
    """
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    while True:
        try:
            # Load new data
            df = load_data(file_path)
            df = preprocess_data(df)
            df = calculate_time_between_orders(df)
            df_filtered = handle_outliers(df)
            X, y = preprocess_features(df_filtered)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict the next order date and notification dates
            df_predictions = predict_next_order_date(model, df_filtered)
            print("\nUpdated Next Order Dates and Notification Dates:")
            print(df_predictions.head())

            # Save predictions to a file
            df_predictions.to_csv(
                "/home/hafsa/Downloads/next_order_dates_with_notifications_date.csv",
                index=False,
            )
            print(
                "\nPredictions and notifications saved to /home/hafsa/Downloads/next_order_dates_with_notifications_date.csv"
            )

        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(86400)  # Sleep for 24 hours


def main():
    """
    Main function to execute the data loading, preprocessing, model training, and predictions.
    """
    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Path to the CSV file
    file_path = "/home/hafsa/Downloads/abc.csv"

    # Load data
    df = load_data(file_path)

    # Display basic info and summary statistics
    print("Basic Info and Summary Statistics:")
    print(df.info())
    print(df.describe())

    # Preprocess data
    df = preprocess_data(df)
    print("\nTime-Based Features Added:")
    print(df[["timestamp", "day_of_week", "season", "total_revenue"]].head())

    # Calculate time between orders
    df = calculate_time_between_orders(df)
    print("\nTime Between Orders Calculated:")
    print(df[["external_customer_id", "timestamp", "time_between_orders"]].head())

    # Handle outliers
    df_filtered = handle_outliers(df)

    # Aggregate customer data
    customer_data = aggregate_customer_data(df_filtered)
    print("\nAggregate Customer Data:")
    print(customer_data.head())

    # Analyze ordering patterns
    order_frequency, preferred_days = analyze_ordering_patterns(df_filtered)
    print("\nOrder Frequency by Customer:")
    print(order_frequency.head())
    print("\nOrder Frequency by Day of Week:")
    print(preferred_days.head())

    # Plot ordering patterns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=preferred_days["day_of_week"], y=preferred_days["order_count"])
    plt.title("Order Frequency by Day of Week")
    plt.savefig("plots/order_frequency_by_day_of_week.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(order_frequency["order_count"], bins=20)
    plt.title("Order Frequency Distribution by Customer")
    plt.savefig("plots/order_frequency_distribution_by_customer.png")
    plt.show()

    # Prepare data for model training
    X, y = preprocess_features(df_filtered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models and display performance metrics
    results = train_models(X_train, y_train, X_test, y_test)
    print("\nModel Performance Metrics:")
    for model_name, metrics in results.items():
        print(f'{model_name}: MSE = {metrics["MSE"]}, MAE = {metrics["MAE"]}')

    # Update predictions periodically
    update_predictions_periodically(file_path)


if __name__ == "__main__":
    main()
