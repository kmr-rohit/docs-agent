FROM python:3.11-slim

WORKDIR /app

COPY feedback-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY feedback-service/app.py feedback-service/db.py ./

ENV PORT=8080
ENV FEEDBACK_DB_PATH=/data/feedback.db

EXPOSE 8080

CMD ["python", "app.py"]
