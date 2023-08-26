import fastapi

app = fastapi.FastAPI()

FEATURES_COLS = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

TARGET_COL = [
    "delay"
]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(flights: dict) -> dict:
    # Verifica si hay columnas desconocidas en los datos
    unknown_columns = [col for col in flights["flights"][0] if col not in FEATURES_COLS + TARGET_COL]

    if unknown_columns:
        unknown_column_names = ", ".join(unknown_columns)
        raise fastapi.HTTPException(status_code=400, detail=f"Unknown column found")
    return {
        "predict": [0]
    }