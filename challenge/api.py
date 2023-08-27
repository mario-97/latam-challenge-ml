import fastapi
from fastapi import HTTPException

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    try:
        flights = data["flights"]
        for flight in flights:
            opera = flight.get("OPERA", "")
            tipo_vuelo = flight.get("TIPOVUELO", "")
            mes = flight.get("MES", 0)

            if not isinstance(opera, str) or not isinstance(tipo_vuelo, str):
                raise fastapi.HTTPException(status_code=400, detail="OPERA and TIPOVUELO must be strings")

            if tipo_vuelo not in ["N", "I"]:
                raise fastapi.HTTPException(status_code=400, detail="TIPOVUELO must be 'N' or 'I'")

            if not (1 <= mes <= 12):
                raise fastapi.HTTPException(status_code=400, detail="Invalid MES value")

        # All validations passed, return the response
        return {"predict": [0]}

    except KeyError:
        raise fastapi.HTTPException(status_code=400, detail="Unknown column found")
