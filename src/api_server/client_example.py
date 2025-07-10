import requests
import json
import time

url = "http://bam-cns-server:8000/analyze"
goal_video_path = "/tmp-video/IMG_5136.mp4"
current_video_path = "/tmp-video/IMG_5136.mp4"

start = time.strftime("%Y-%m-%d %H:%M:%S")

goal_frame_idx = 25
current_frame_idx = 0

results = []

while current_frame_idx < 100:
    with open(goal_video_path, "rb") as f1, open(current_video_path, "rb") as f2:
        files = {
            "goal_video": ("goal.mp4", f1, "video/mp4"),
            "current_video": ("current.mp4", f2, "video/mp4"),
        }
        params = {"goal_frame_idx": goal_frame_idx, "current_frame_idx": current_frame_idx, "id": start}
        response = requests.post(url, files=files, params=params)

    print(f"[{goal_frame_idx}] {response.status_code}")

    try:
        response_data = response.json()
        response_data["goal_frame_idx"] = goal_frame_idx
        response_data["current_frame_idx"] = current_frame_idx
        results.append(response_data)
    except Exception:
        print(f"Failed to parse JSON for frame {goal_frame_idx}: {response.text}")

    current_frame_idx += 1
    # goal_frame_idx += 1

with open("metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved metrics.json with all responses.")