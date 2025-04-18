# server/main.py
import io, base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt

app = FastAPI()

# Allow your React dev server (http://localhost:5173) to call these APIs
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is alive!"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    data = await file.read()
    df = pd.read_csv(io.BytesIO(data))

    # example cleaning
    df = df.dropna()
    df.columns = df.columns.str.strip().str.lower()

    return {
      "columns": df.columns.tolist(),
      "head":    df.head().to_dict(orient="records")
    }

@app.get("/plot")
def get_plot():
    plt.figure()
    pd.Series([1, 3, 2, 5]).plot(kind="bar")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return {"plot": f"data:image/png;base64,{img_b64}"}
