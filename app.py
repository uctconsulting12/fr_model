from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException

from src.websocket.FR_detection_websocket import run_FR_detection

from src.handlers.FR_handler import fr_websocket_handler

from fastapi.middleware.cors import CORSMiddleware


from concurrent.futures import ThreadPoolExecutor

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



