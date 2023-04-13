FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5003
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
