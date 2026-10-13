FROM python:3.13-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only the code and artifacts the API needs; data and notebooks stay out of the image
COPY src/ src/
COPY app/ app/
COPY models/logistic_regression.joblib models/serving_config.json models/shap_background.csv models/

# Prediction log lives on a volume so it survives container restarts
ENV PREDICTIONS_DB=/data/predictions.db
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
