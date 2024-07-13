import pandas as pd
from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.pipeline import PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
pipeline = PredictPipeline()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dataset_predict", response_class=HTMLResponse)
def render_dataset_form(request: Request):
    return templates.TemplateResponse("dataset_predict.html", {"request": request})

@app.post("/dataset_predict", response_class=HTMLResponse)
def predict_dataset(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        processed_df = pipeline.preprocess_dataset(df)
        predictions = pipeline.predict(processed_df)
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "predicted_classes": predictions})
    except ValueError as e:
        return templates.TemplateResponse("dataset_predict.html", {"request": request, "error_message": f"Error processing dataset: {e}"})