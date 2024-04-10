from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse

import joblib

app = FastAPI(
    title="Trabajo 7",
    description="Fabiola Díaz y Valerie Espinoza",
    version="0.0.1"
)

model, scaler = joblib.load("model_v1.pkl")


@app.post("/api/v1/predict-tesla-stock", tags=["tesla"])
async def predict(
        High: float,
        Low: float,
        Open: float
):
    try:
        if not all(isinstance(value, (int, float)) for value in [High, Low, Open]):
            raise HTTPException(
                detail="Los valores de entrada deben ser numéricos",
                status_code=status.HTTP_400_BAD_REQUEST
            )

        scaled_data = scaler.transform([[High, Low, Open]])

        prediction = model.predict(scaled_data)
        prediction_value = prediction[0]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": prediction_value}
        )

    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )
