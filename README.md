
# FastAPI Project

This project uses FastAPI to build  API's with background processing and machine learning functionalities. It processes customer order data, predicts future order dates and notification dates, and sends email notifications with the results. 

## Project Structure

```
projectname/
│
├── data/                            # Directory for CSV data files
│   └── abc.csv                      # Example data file
│
├── pythonProject7/                  # Project source directory
│   ├── __init__.py
│   ├── ML.py                        # Machine learning utility functions
│   ├── main.py                      # FastAPI application entry point
│   ├── main_bearer.py               # JWT bearer authentication
│   ├── main_handler.py              # Token creation utility                
│   ├── requirements.txt             # Project dependencies
├── .env                             # Environment variables
├── .venv                            # Virtual environment directory
└── README.md                        # This file
```

## Code Explanation

### `ML.py`

- **Imports and Setup**: Imports necessary libraries and sets up directories for data and plots.
- **`load_data(file_name)`**: Loads a CSV file into a DataFrame.
- **`preprocess_data(df)`**: Preprocesses the data by handling missing values, converting datatypes, and creating new features.
- **`calculate_time_between_orders(df)`**: Calculates the time between orders for each customer.
- **`aggregate_customer_data(df)`**: Aggregates data at the customer level for analysis.
- **`analyze_ordering_patterns(df)`**: Analyzes ordering patterns, including order frequency and preferred order days.
- **`handle_outliers(df)`**: Visualizes and removes outliers in the 'price' column.
- **`preprocess_features(df)`**: Preprocesses features for model training, including handling categorical and numerical features.
- **`train_models(x_train, y_train, x_test, y_test)`**: Trains various models (Linear Regression, Random Forest, SVR, XGBoost) and evaluates their performance.
- **`predict_next_order_date(model, df, days_before_notification=3)`**: Predicts the next order date and calculates notification dates.
- **`update_predictions_periodically(file_path)`**: Periodically updates predictions and saves them to a CSV file.
- **`main()`**: The main function that orchestrates data loading, preprocessing, model training, and predictions.

### `main_bearer.py`

- **`JWTBearer` Class**: Custom HTTPBearer security scheme for JWT authentication.
  - **`__call__()`**: Extracts and verifies the JWT token from the request.
  - **`verify_jwt(jwtoken)`**: Verifies the validity of the JWT token.

### `main_handler.py`

- **Imports**: Imports necessary libraries and loads environment variables.
- **`create_access_token(data, expires_delta)`**: Creates a JWT access token with an optional expiration time.
- **`decode_jwt(token, credentials_exception=None)`**: Decodes and validates a JWT token.
- **`get_current_user(token: str = Depends(oauth2_scheme))`**: Retrieves the current user from the JWT token.

### `main.py`

- **Imports**: Imports necessary libraries and modules, including FastAPI, APScheduler, and machine learning functions.
- **`process_and_send_email()`**: Processes data, makes predictions, and sends an email with the results using `yagmail`.
- **`scheduler.add_job()`**: Adds a background job to process data and send an email every 24 hours.
- **FastAPI Endpoints**:
  - **`GET /`**: Returns a message indicating that the API is running. Requires JWT authentication.
  - **`POST /start-processing/`**: Starts data processing and sends an email immediately. Requires JWT authentication.
  - **`GET /token`**: Generates a JWT token for testing purposes.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd projectname
    ```

2. **Set up the virtual environment**:
    ```bash
    python -m venv .venv
    ```

3. **Activate the virtual environment**:
    - **Linux/MacOS**:
      ```bash
      source .venv/bin/activate
      ```

4. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Set up environment variables**:
   Create a `.env` file in the root directory and add the following variables:
    ```env
    EMAIL_USER=your-email@example.com
    EMAIL_PASSWORD=your-email-password
    RECIPIENT_EMAIL=recipient@example.com
    SECRET_KEY=your-secret-key
    ```

## Usage

1. **Start the FastAPI application**:
    ```bash
    uvicorn pythonProject7.main:app --reload
    ```

2. **Access the API**:
    - **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    - **OpenAPI JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

3. **Endpoints**:
    - **GET `/`**: Returns a message indicating that the API is running.
    - **POST `/start-processing/`**: Starts processing data and sends an email immediately. Requires JWT authentication.
    - **GET `/token`**: Generates a JWT token for testing.

## Automated Tasks

The application uses APScheduler to run a background job every 24 hours. This job processes data, generates predictions, and sends an email with the results.

## Development

1. **Run tests** (if applicable):
    ```bash
    pytest
    ```

2. **Format code**:
    - **Black**:
      ```bash
      black .
      ```
    - **isort**:
      ```bash
      isort .
      ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
