from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from core_ml import check_liveness

app = FastAPI(
    title="The Digital Bouncer Liveness API",
    description="Microservice for Selfie Liveness Verification",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Liveness MLOps Backend"}

@app.post("/verify")
async def verify_identity(selfie: UploadFile = File(...)):
    try:
        # Read the selfie file bytes into memory
        selfie_bytes = await selfie.read()

        # Pass bytes to your Liveness CNN
        liveness_score = check_liveness(selfie_bytes)

        # Approve if liveness is above 70%
        is_passed = liveness_score > 0.70

        return {
            "status": "APPROVED" if is_passed else "REJECTED",
            "liveness_score": round(liveness_score, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)