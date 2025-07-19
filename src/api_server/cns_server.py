"""
FastAPI server for CNS video analysis.
Receives video files and performs velocity analysis using CNS pipeline.
"""


import sys
import os
from enum import Enum
import shutil
import tempfile
import traceback
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import cv2
from codecarbon import EmissionsTracker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cns_external_images import run_cns_with_external_images
from utils.veryutils import create_zip_stream, convert_ndarray
from pydantic import BaseModel
from typing import List

class CNSResponseModel(BaseModel):
    request_id: str
    velocity: List[List[float]]
    carbon_footprint: float
    download_url: str
    message: str
    

class DetectorEnum(str, Enum):
    akaze = "AKAZE"
    sift = "SIFT"
    orb = "ORB"
    
semaphore = asyncio.Semaphore(2)

app = FastAPI(
    title="CNS Video Analysis API",
    description="API for analyzing video frames using CNS pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        

def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=400, detail=f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames
   

def extract_frame(video_path: str, frame_idx: int):
    """Extract a specific frame from a video file."""
    cap = None
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail=f"Cannot open video file: {video_path}"
            )

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
            
            
@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "message": "CNS server is running"
    })
            

@app.post("/analyze", response_model=CNSResponseModel)
async def analyze(
    jobId: str = Form("untitled", description="Job identifier from Bull"),
    device: str = Form("cuda:0", description="Device to run CNS on: 'cuda:0' or 'cpu'"),
    detector: DetectorEnum = Form("AKAZE", description="Feature extractor algorithm: AKAZE, SIFT or ORB"),
    goal_video: UploadFile = File(..., description="Goal video file"),
    current_video: Optional[UploadFile] = File(None, description="Current video file (optional)"),
    goal_frame_idx: int = Form(0, ge=0, description="Frame index to extract from goal video"),
    frame_step: int = Form(1, ge=1, description="Sampling rate for processing"),
    start_frame: int = Form(0, ge=0, description="Frame to start the process"),
    end_frame: int = Form(0, ge=1, description="Frame to finish the process")
):
    """
    Analyze video frames using CNS pipeline.

    Args:
        jobId: Job identifier from Bull (default: "untitled")
        device: Device to run CNS on (default cuda:0)
        detector: Feature extractor algorithm: AKAZE, SIFT or ORB (default AKAZE)
        goal_video: The goal video file (required)
        current_video: The current video file (optional, will use goal_video if not provided)
        goal_frame_idx: Frame index for goal video (default: 0)
        frame_step: Sampling rate for processing (default: 1)
        start_frame: Frame to start the process (default: 0)
        end_frame: Frame to finish the process (default: 0)

    Returns:
        JSON response with velocity and carbon footprint data
    """
    
    async with semaphore:

        # Initialize emissions tracker
        tracker = EmissionsTracker(
            save_to_file = False
        )
        tracker_started = False
        try:
            tracker.start()
            tracker_started = True
        except Exception as e:
            print(f"[WARNING] Could not start EmissionsTracker: {e}")

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
                
                # Save current video to temp file first
                curr_path = save_upload_to_tempfile(current_video)
                current_total_frames = get_total_frames(curr_path)
            else:
                # Use goal video as current video
                goal_path = save_upload_to_tempfile(goal_video)
                current_total_frames = get_total_frames(goal_path)
                curr_path = goal_path
                    
            if start_frame >= end_frame:
                raise HTTPException(
                    status_code=400,
                    detail=f"'start_frame' ({start_frame}) must be less than 'end_frame' ({end_frame})"
                )

            if end_frame > current_total_frames:
                raise HTTPException(
                    status_code=400,
                    detail=f"'end_frame' ({end_frame}) exceeds video frame count for current video ({current_total_frames})"
                )

            print(f"[INFO] Processing request ID: {jobId}")
            print(f"[INFO] Goal video: {goal_video.filename} (frame {goal_frame_idx})")

            # Save goal video to temporary file
            goal_path = save_upload_to_tempfile(goal_video)
            print(f"[DEBUG] Goal video saved to: {goal_path}")

            # Handle current video (curr_path already set above)
            if current_video and current_video.filename:
                print(f"[DEBUG] Current video saved to: {curr_path}")
            else:
                print(f"[DEBUG] Using goal video as current video")

            goal_img = extract_frame(goal_path, goal_frame_idx)

            print(f"[INFO] Running CNS analysis with detector {detector} on device: {device}")
            for current_frame_idx in range(start_frame, end_frame, frame_step):
                # Extract frames from current video
                print(f"[INFO] Extracting frames...")
                current_img = extract_frame(curr_path, current_frame_idx)
                if goal_img is None or current_img is None:
                    raise HTTPException(status_code=400, detail="Failed to extract valid frames")
                print(f"[INFO] Frames extracted successfully")

                # Run CNS analysis
                vel = run_cns_with_external_images(
                    goal_img=goal_img,
                    current_img=current_img,
                    device=device,
                    detector=detector,
                    id=jobId,
                    frame_idx=(goal_frame_idx, current_frame_idx)
                )

                if vel is None:
                    raise HTTPException(
                        status_code=500, detail="CNS pipeline failed to produce results")

                # Convert velocity to list format for JSON serialization
                velocity = convert_ndarray(vel)
                response_data.append({
                    "velocity": velocity,
                })

            # Prepare response (emissions will be added in finally)
            print(f"[INFO] CNS Pipeline completed successfully")

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error in analysis: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}")

        finally:
            emissions = None
            if 'tracker' in locals() and tracker_started:
                try:
                    emissions = tracker.stop()      
                    print(
                        f"[INFO] Carbon emissions for request {jobId}: {float(emissions):.6f} kg CO₂eq")
                except Exception as e:
                    print(f"[WARNING] Could not stop EmissionsTracker: {e}")

            if emissions is not None:
                format_emissions = round(float(emissions), 6)
                response_data.append({
                    "request_id": jobId,
                    "carbon_footprint": format_emissions
                })
                
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
                    
            velocities = [item["velocity"]
                          for item in response_data if "velocity" in item]

            structured_response = {
                "requestId": jobId,
                "velocity": velocities,
                "carbon_footprint": round(float(emissions), 6) if emissions else 0,
                "download_url": f"/download/{jobId}",
                "message": "Use this URL to download the generated visualization images as a ZIP file"
            }

            return JSONResponse(content=structured_response)


@app.get("/download/{request_id}")
async def download_images(request_id: str):
    """
    Download visualization images for a specific request as a ZIP file.
    
    Args:
        request_id: The request ID to download images for
        
    Returns:
        StreamingResponse: ZIP file containing all visualization images
    """
    try:
        # Create ZIP archive in memory and clean up files
        zip_bytes = create_zip_stream(request_id, cleanup_after=True)
        
        # Create BytesIO buffer and return as streaming response
        zip_buffer = io.BytesIO(zip_bytes)
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={request_id}_images.zip"}
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"No images found for request ID: {request_id}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404, 
            detail=str(e)
        )
    except Exception as e:
        print(f"[ERROR] Error creating ZIP download: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


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
        host="0.0.0.0",
        port=8000,
    )
