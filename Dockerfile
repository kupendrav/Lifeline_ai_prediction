FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV FLASK_ENV=production
EXPOSE 5000
CMD ["sh", "-c", "gunicorn run:app --workers 2 --bind 0.0.0.0:${PORT:-5000} --timeout 60"]
