import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
import argparse
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cns_external_images import run_cns_with_external_images

app = FastAPI()

DEVICE = None

def save_upload_to_tempfile(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name

def extract_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=400, detail=f"Cannot extract frame {frame_idx} from {video_path}")
    return frame

@app.post("/analyze")
def analyze(
    goal_video: UploadFile = File(...),
    current_video: UploadFile = File(None),  # optional second video
    goal_frame_idx: int = Query(0, ge=0),
    current_frame_idx: int = Query(0, ge=0),
    id: str = "untitled"
):
    goal_path = save_upload_to_tempfile(goal_video)

    if current_video is not None:
        curr_path = save_upload_to_tempfile(current_video)
    else:
        curr_path = goal_path  # same video for current

    try:
        goal_img = extract_frame(goal_path, goal_frame_idx)
        current_img = extract_frame(curr_path, current_frame_idx)

        if goal_img is None or current_img is None:
            raise HTTPException(status_code=400, detail="Invalid image extracted")

        vel, data, timing = run_cns_with_external_images(goal_img=goal_img, current_img=current_img, device=DEVICE, id=id, frame_idx=(goal_frame_idx, current_frame_idx))

        if vel is None:
            raise HTTPException(status_code=400, detail="CNS pipeline failed")

        return JSONResponse({
            "velocity": vel.tolist() if hasattr(vel, "tolist") else vel,
            "timing": timing,
            "data": str(data)
        })

    finally:
        if os.path.exists(goal_path):
            os.remove(goal_path)
        if current_video is not None and os.path.exists(curr_path):
            os.remove(curr_path)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="Device to run CNS on: 'cuda:0' or 'cpu'")
    args = parser.parse_args()
    DEVICE = args.device
    print(f"[INFO] CNS device set to: {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)