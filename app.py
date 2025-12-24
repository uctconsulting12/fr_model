from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from src.websocket.FR_detection_websocket import run_FR_detection

from src.handlers.FR_handler import fr_websocket_handler

from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import logging  
from src.websocket.add_employee import  register_face
import os
import shutil
import uuid
from typing import Optional, Dict, Any

logger = logging.getLogger("face_registration")
logging.basicConfig(level=logging.INFO)


UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores ----------------
fr_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- PPE WebSocket ----------------
@app.websocket("/ws/FR/{client_id}")
async def websocket_ppe(ws: WebSocket,client_id: str):
    await fr_websocket_handler(detection_executor, storage_executor, ws,client_id, fr_sessions, run_FR_detection, "FR_Detection")



# -----------------------------
# Request schema
# -----------------------------
class EmployeeRegisterRequest(BaseModel):
    emp_id: int
    image_path: str
    name: str
    department: str
    org_id: int
    user_id: int
    email: str


# -----------------------------
# Response schema
# -----------------------------
class RegisterResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None






# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/register-face", response_model=RegisterResponse)
async def register_face_api(
    emp_id: int = Form(...),
    name: str = Form(...),
    department: str = Form(...),
    org_id: int = Form(...),
    user_id: int = Form(...),
    email: str = Form(...),
    image: UploadFile = File(...)
):
    temp_path = None

    try:
        # Validate image
        if image.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(
                status_code=400,
                detail="Only JPG and PNG images allowed"
            )

        # Save temporarily
        ext = image.filename.split(".")[-1]
        temp_path = os.path.join(
            UPLOAD_DIR, f"{uuid.uuid4()}.{ext}"
        )

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        

        result = register_face(
            emp_id,
            temp_path,
            name,
            department,
            org_id,
            user_id,
            email=""
        )

        # Determine success
        is_success = result.get("status") == "success"

        return RegisterResponse(
            success=is_success,
            message=result.get("message", "Face registration failed"),
            data=result if not is_success else None
        )

    

    except Exception as e:
        logger.exception("Face registration error")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # ALWAYS delete temp image
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Deleted temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")