FROM python:3.12
WORKDIR /air_system_pred
COPY /requirements.txt /air_system_pred/requirements.txt
RUN pip install -r /air_system_pred/requirements.txt
COPY /app /air_system_pred/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]