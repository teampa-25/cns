#!/usr/bin/env python3
"""
Simple video client for CNS API server.
Sends video files to the server and receives velocity analysis results.
"""

import argparse
import requests
import json
import os
import time
from typing import Optional, Dict, Any


class CNSVideoClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        """Initialize the CNS video client.
        
        Args:
            server_url: Base URL of the CNS API server
        """
        self.server_url = server_url.rstrip('/')
        self.analyze_endpoint = f"{self.server_url}/analyze"
    
    def send_video(
        self,
        goal_video_path: str,
        current_video_path: Optional[str] = None,
        goal_frame_idx: int = 0,
        current_frame_idx: int = 0,
        sampling_rate: int = 1,
        request_id: Optional[str] = None
    ) -> Dict[Any, Any]:
        """Send video(s) to the CNS server for analysis.
        
        Args:
            goal_video_path: Path to the goal video file
            current_video_path: Path to the current video file (optional)
            goal_frame_idx: Frame index to extract from goal video
            current_frame_idx: Frame index to extract from current video
            sampling_rate: Sampling rate for processing
            request_id: Unique identifier for this request
            
        Returns:
            Dictionary containing the server response
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            requests.RequestException: If server request fails
        """
        # Validate video files exist
        if not os.path.exists(goal_video_path):
            raise FileNotFoundError(f"Goal video not found: {goal_video_path}")
        
        if current_video_path and not os.path.exists(current_video_path):
            raise FileNotFoundError(f"Current video not found: {current_video_path}")
        
        # Prepare request parameters
        params = {
            "goal_frame_idx": goal_frame_idx,
            "current_frame_idx": current_frame_idx,
            "sampling_rate": sampling_rate,
            "id": request_id or f"request_{int(time.time())}"
        }
        
        # Prepare files for upload
        files = {}
        
        try:
            # Always include goal video
            with open(goal_video_path, "rb") as goal_file:
                files["goal_video"] = (
                    os.path.basename(goal_video_path),
                    goal_file.read(),
                    "video/mp4"
                )
                
                # Include current video if provided
                if current_video_path:
                    with open(current_video_path, "rb") as current_file:
                        files["current_video"] = (
                            os.path.basename(current_video_path),
                            current_file.read(),
                            "video/mp4"
                        )
                
                # Send request to server
                print(f"Sending request to {self.analyze_endpoint}")
                print(f"Goal video: {goal_video_path} (frame {goal_frame_idx})")
                if current_video_path:
                    print(f"Current video: {current_video_path} (frame {current_frame_idx})")
                else:
                    print(f"Current video: same as goal (frame {current_frame_idx})")
                
                response = requests.post(
                    self.analyze_endpoint,
                    files=files,
                    params=params,
                    timeout=30
                )
                
                # Check response status
                response.raise_for_status()
                
                # Parse and return JSON response
                result = response.json()
                print(f"✓ Analysis completed successfully")
                return result
                
        except requests.RequestException as e:
            print(f"✗ Server request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse server response as JSON: {e}")
            print(f"Raw response: {response.text}")
            raise
    
    def send_video_sequence(
        self,
        goal_video_path: str,
        current_video_path: Optional[str] = None,
        goal_frame_idx: int = 0,
        start_frame: int = 0,
        end_frame: int = 100,
        frame_step: int = 1,
        sampling_rate: int = 1,
        output_file: Optional[str] = None
    ) -> list:
        """Send multiple frames from video(s) for sequential analysis.
        
        Args:
            goal_video_path: Path to the goal video file
            current_video_path: Path to the current video file (optional)
            goal_frame_idx: Frame index for the goal (fixed)
            start_frame: Starting frame index for current video
            end_frame: Ending frame index for current video
            frame_step: Step between frames
            sampling_rate: Sampling rate for processing
            output_file: Optional file to save results as JSON
            
        Returns:
            List of analysis results for each frame
        """
        results = []
        request_id = f"sequence_{int(time.time())}"
        
        print(f"Processing frame sequence: {start_frame} to {end_frame-1} (step: {frame_step})")
        
        for current_frame_idx in range(start_frame, end_frame, frame_step):
            try:
                result = self.send_video(
                    goal_video_path=goal_video_path,
                    current_video_path=current_video_path,
                    goal_frame_idx=goal_frame_idx,
                    current_frame_idx=current_frame_idx,
                    sampling_rate=sampling_rate,
                    request_id=f"{request_id}_frame_{current_frame_idx}"
                )
                
                # Add frame indices to result for tracking
                result["goal_frame_idx"] = goal_frame_idx
                result["current_frame_idx"] = current_frame_idx
                results.append(result)
                
                print(f"Frame {current_frame_idx}: velocity = {result.get('velocity', 'N/A')}")
                
            except Exception as e:
                print(f"✗ Failed to process frame {current_frame_idx}: {e}")
                continue
        
        # Save results if output file specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="CNS Video Client - Send videos to CNS API server")
    parser.add_argument("goal_video", help="Path to goal video file")
    parser.add_argument("--current-video", help="Path to current video file (optional)")
    parser.add_argument("--server-url", default="http://localhost:8000", help="CNS server URL")
    parser.add_argument("--goal-frame", type=int, default=0, help="Goal frame index")
    parser.add_argument("--current-frame", type=int, default=0, help="Current frame index")
    parser.add_argument("--sampling-rate", type=int, default=1, help="Sampling rate")
    parser.add_argument("--sequence", action="store_true", help="Process frame sequence")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame for sequence")
    parser.add_argument("--end-frame", type=int, default=100, help="End frame for sequence")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step for sequence")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--id", help="Request ID")
    
    args = parser.parse_args()
    
    # Create client
    client = CNSVideoClient(args.server_url)
    
    try:
        if args.sequence:
            # Process frame sequence
            results = client.send_video_sequence(
                goal_video_path=args.goal_video,
                current_video_path=args.current_video,
                goal_frame_idx=args.goal_frame,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                frame_step=args.frame_step,
                sampling_rate=args.sampling_rate,
                output_file=args.output
            )
            print(f"\nProcessed {len(results)} frames successfully")
        else:
            # Process single frame pair
            result = client.send_video(
                goal_video_path=args.goal_video,
                current_video_path=args.current_video,
                goal_frame_idx=args.goal_frame,
                current_frame_idx=args.current_frame,
                sampling_rate=args.sampling_rate,
                request_id=args.id
            )
            
            print("\nAnalysis Result:")
            print(f"Velocity: {result.get('velocity', 'N/A')}")
            print(f"Timing: {result.get('timing', 'N/A')}")
            print(f"Carbon footprint: {result.get('carbon_footprint', 'N/A')}")
            
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
                
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
