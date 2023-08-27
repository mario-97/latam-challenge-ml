import fastapi

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return  {
        "predict": [0]
    }

""" @app.post("/predict", status_code=200)
async def post_predict(flights: dict) -> dict:
    # Verifica si hay columnas desconocidas en los datos
    unknown_columns = [col for col in flights["flights"][0] if col not in FEATURES_COLS + TARGET_COL]

    if unknown_columns: 
        raise fastapi.HTTPException(status_code=400, detail=f"Unknown column found")
    return {
        "predict": [0]
    } """