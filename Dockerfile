FROM python:3.13-slim
WORKDIR /app

COPY flask_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app/ /app/flask_app/
COPY artifacts/scaler.pkl /app/artifacts/scaler.pkl

WORKDIR /app/flask_app
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]