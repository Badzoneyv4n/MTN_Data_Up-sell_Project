from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import sys
from io import StringIO

# Add the parent directory's 'scripts' folder to the Python path
sys.path.append(os.path.abspath('../scripts'))

# from recommender import recommend

from scripts.recommender import recommend

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def main_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/users", response_class=HTMLResponse)
async def recommend_users(
    request: Request,
    file: UploadFile = File(...)
):
    # Handle CSV upload for multiple users
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode()))
    result = recommend(df)
    results = result.to_dict(orient="records")
    return templates.TemplateResponse("result.html", {"request": request, "results": results})

@app.post("/direct", response_class=HTMLResponse)
async def recommend_direct(
    request: Request,
    avg_data_before_upgrade: float = Form(...),
    std_before: float = Form(...),
    total_recharge_before: float = Form(...),
    data_flag: int = Form(...),
    std_flag: int = Form(...),
    recharge_flag: int = Form(...)
):
    # Handle direct input for a single user
    df = pd.DataFrame([{
        "avg_data_before_upgrade": avg_data_before_upgrade,
        "std_before": std_before,
        "total_recharge_before": total_recharge_before,
        "data_flag": data_flag,
        "std_flag": std_flag,
        "recharge_flag": recharge_flag
    }])

    print(df)
    result = recommend(df, mode='direct')
    results = result.to_dict(orient="records")
    return templates.TemplateResponse("result.html", {"request": request, "results": results})