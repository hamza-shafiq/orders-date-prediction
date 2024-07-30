import os
import yagmail
from fastapi import FastAPI, BackgroundTasks, Depends
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pythonProject7.main_bearer import JWTBearer
from pythonProject7.main_handler import create_access_token
from pythonProject7.ML import load_data, preprocess_data, calculate_time_between_orders, handle_outliers, preprocess_features, predict_next_order_date



load_dotenv()

app = FastAPI()

scheduler = BackgroundScheduler()
scheduler.start()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, "..", "data", "abc.csv")
PREDICTIONS_FILE_PATH = os.path.join(BASE_DIR, "..", "next_order_dates_with_notifications_date.csv")

def process_and_send_email():
    """
    Process data, save predictions, and send email.
    """
    try:
        df = load_data(DATA_FILE_PATH)
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
        df_predictions.to_csv(PREDICTIONS_FILE_PATH, index=False)

        yag.send(
            to=RECIPIENT_EMAIL,
            subject="Updated Next Order Dates and Notifications",
            contents="Please find attached the updated next order dates and notifications.",
            attachments=PREDICTIONS_FILE_PATH,
        )
        print(f"Email sent with the file: {PREDICTIONS_FILE_PATH}")

    except Exception as e:
        print(f"An error occurred during processing or sending email: {e}")

scheduler.add_job(
    process_and_send_email,
    trigger=IntervalTrigger(hours=24),
    id="process_and_send_email_job",
    name="Process data and send email every 24 hours",
    replace_existing=True,
)

@app.get("/", dependencies=[Depends(JWTBearer())])
def read_root():
    return {"message": "API is running"}

@app.post("/start-processing/", dependencies=[Depends(JWTBearer())])
def start_processing(background_tasks: BackgroundTasks):
    """
    Start processing data and send email immediately.
    """
    background_tasks.add_task(process_and_send_email)
    return {"message": "EMAIL SENT :)"}

@app.get("/token", response_model=str)
def get_token():
    """
    Generate a token
    """
    token = create_access_token(data={"sub": "testuser"})
    return token

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
