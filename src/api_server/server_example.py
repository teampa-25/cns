
"""
FastAPI server for CNS video analysis.
Receives video files and performs velocity analysis using CNS pipeline.
"""

from enum import Enum
import os
import shutil
import tempfile
import traceback
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import sys
import cv2
from codecarbon import EmissionsTracker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cns_external_images import run_cns_with_external_images


class DetectorEnum(str, Enum):
    akaze = "AKAZE"
    sift = "SIFT"
    orb = "ORB"


app = FastAPI(
    title="CNS Video Analysis API",
    description="API for analyzing video frames using CNS pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def save_upload_to_tempfile(upload_file: UploadFile) -> str:
    """Save uploaded file to a temporary file and return the path."""
    try:
        # Create temporary file with proper extension
        file_extension = os.path.splitext(upload_file.filename)[
            1] if upload_file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving uploaded file: {str(e)}")


def extract_frame(video_path: str, frame_idx: int):
    """Extract a specific frame from a video file."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, detail=f"Cannot open video file: {video_path}")

        # Get total frame count for validation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= total_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Frame index {frame_idx} exceeds video length ({total_frames} frames)"
            )

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot extract frame {frame_idx} from video"
            )

        return frame

    except cv2.error as e:
        raise HTTPException(status_code=400, detail=f"OpenCV error: {str(e)}")
    except Exception as e:
        if "Frame index" in str(e) or "Cannot extract frame" in str(e):
            raise  # Re-raise HTTP exceptions
        raise HTTPException(
            status_code=500, detail=f"Error extracting frame: {str(e)}")
    finally:
        if cap is not None:
            cap.release()


@app.post("/analyze")
async def analyze(
    id: str = Form("untitled", description="Request identifier"),
    device: str = Form("cuda:0", description="Device to run CNS on: 'cuda:0' or 'cpu'"),
    detector: DetectorEnum = Form("AKAZE", description="Feature extractor algorithm: AKAZE, SIFT or ORB"),
    goal_video: UploadFile = File(..., description="Goal video file"),
    current_video: Optional[UploadFile] = File(None, description="Current video file (optional)"),
    goal_frame_idx: int = Form(0, ge=0, description="Frame index to extract from goal video"),
    frame_step: int = Form(1, ge=1, description="Sampling rate for processing"),
    start_frame: int = Form(0, ge=0, description="Frame to start the process"),
    end_frame: int = Form(0, ge=100, description="Frame to finish the process")
):
    """
    Analyze video frames using CNS pipeline.

    Args:
        id: Request identifier (default: "untitled")
        device: Device to run CNS on (default cuda:0)
        detector: Feature extractor algorithm: AKAZE, SIFT or ORB (default AKAZE)
        goal_video: The goal video file (required)
        current_video: The current video file (optional, will use goal_video if not provided)
        goal_frame_idx: Frame index for goal video (default: 0)
        frame_step: Sampling rate for processing (default: 1)
        start_frame: Frame to start the process (default: 0)
        end_frame: Frame to finish the process (default: 100)


    Returns:
        JSON response with velocity and carbon footprint data
    """

    # Initialize emissions tracker
    tracker = EmissionsTracker()
    tracker.start()

    goal_path = None
    curr_path = None
    current_frame_idx = 0
    response_data = [{}]

    try:
        # Validate uploaded files
        if not goal_video.filename:
            raise HTTPException(
                status_code=400, detail="Goal video filename is required")

        # Check file types
        allowed_extension = '.mp4'
        goal_ext = os.path.splitext(goal_video.filename)[1].lower()
        if goal_ext not in allowed_extension:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {goal_ext}. Supported formats: {allowed_extension}"
            )

        if current_video and current_video.filename:
            curr_ext = os.path.splitext(current_video.filename)[1].lower()
            if curr_ext not in allowed_extension:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {curr_ext}. Supported formats: {allowed_extension}"
                )

        print(f"[INFO] Processing request ID: {id}")
        print(
            f"[INFO] Goal video: {goal_video.filename} (frame {goal_frame_idx})")

        # Save goal video to temporary file
        goal_path = save_upload_to_tempfile(goal_video)
        print(f"[INFO] Goal video saved to: {goal_path}")

        # Handle current video
        if current_video and current_video.filename:
            curr_path = save_upload_to_tempfile(current_video)
            print(f"[INFO] Current video saved to: {curr_path}")
        else:
            curr_path = goal_path  # Use same video for current
            print(
                f"[INFO] Using goal video as current video (frame {current_frame_idx})")

        goal_img = extract_frame(goal_path, goal_frame_idx)

        for current_frame_idx in range(start_frame, end_frame, frame_step):
            # Extract frames from current video
            print(f"[INFO] Extracting frames...")
            current_img = extract_frame(curr_path, current_frame_idx)
            if goal_img is None or current_img is None:
                raise HTTPException(
                    status_code=400, detail="Failed to extract valid frames")
            print(f"[INFO] Frames extracted successfully")

            # Run CNS analysis
            print(f"[INFO] Running CNS analysis on device: {device}")
            vel = run_cns_with_external_images(
                goal_img=goal_img,
                current_img=current_img,
                device=device,
                detector=detector,
                id=id,
                frame_idx=(goal_frame_idx, current_frame_idx)
            )

            if vel is None:
                raise HTTPException(
                    status_code=500, detail="CNS pipeline failed to produce results")

            response_data.append({
                "velocity": vel.tolist() if hasattr(vel, "tolist") else vel,
            })

        # Stop emissions tracking
        emissions = tracker.stop()
        print(f"[INFO] Analysis completed. Emissions: {emissions}")

        # Prepare response
        response_data.append({
            "request_id": id,
            "carbon_footprint": emissions
        })

        print(f"[INFO] Request {id} completed successfully")
        return JSONResponse(content=response_data)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in analysis: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Clean up temporary files
        cleanup_files = []
        if goal_path and os.path.exists(goal_path):
            cleanup_files.append(goal_path)
        if curr_path and curr_path != goal_path and os.path.exists(curr_path):
            cleanup_files.append(curr_path)

        for file_path in cleanup_files:
            try:
                os.remove(file_path)
                print(f"[INFO] Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up {file_path}: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    print(f"[ERROR] Unhandled exception: {str(exc)}")
    print(f"[ERROR] Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port="8000",
    )
