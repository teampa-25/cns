
# FastCNS: not your ordinary API
<div align="center">
  <img width="220" height="240" alt="CNS Banner" src="https://github.com/user-attachments/assets/b0dad17b-9bde-474a-817a-c9062f2aa987" />
</div>

## Overview
A modular, GPU-accelerated system developed ad-hoc for visual servoing. It provides a FastAPI server for high-performance inference, supports custom pipelines, and is ready for both research and production environments. Note that it is also provided the training script.

## Features
- **FastAPI server** for easy integration and RESTful access
- **GPU acceleration** (CUDA, PyTorch, Torch Geometric, OpenCV)
- **Automatic carbon footprint tracking** (via CodeCarbon)
- **Rich visualization**: keypoints, matches, graphs, and more
- **Dockerized**: run anywhere, reproducible environments
- **Streaming ZIP download** of result images
- **Extensive configuration** via JSON and API parameters

## Quick Start
1. **Build the Docker Image**
   ```sh
   docker build -t cns-powa .
   ```
2. **Run the Container** (with GPU, volume, and port mapping)
   ```powershell
   $repoPath = (Get-Location).Path
   docker run --gpus all -it -v "${repoPath}:/workspace" --workdir /workspace -p 8000:8000 cns-powa
   ```
3. **Start the API Server**
   Inside the container (with the correct venv activated):
   ```sh
   python src/api_server/cns_server.py --device cuda:0
   ```

## API Usage

### Analyze Video

**POST** `/analyze`

- Upload `goal_video` (required) and `current_video` (optional)
- Parameters: `device`, `detector`, `goal_frame_idx`, `frame_step`, `start_frame`, `end_frame`, etc.

**Example (Python):**
```python
import requests

files = {
    'goal_video': open('data/goal.mp4', 'rb'),
    'current_video': open('data/current.mp4', 'rb')
}
data = {
    'device': 'cuda:0',
    'detector': 'AKAZE',
    'goal_frame_idx': 0,
    'start_frame': 0,
    'end_frame': 100
}
r = requests.post('http://localhost:8000/analyze', files=files, data=data)
print(r.json())
```

### Download Results

After analysis, the response includes a `download_url` (e.g. `/download/your_job_id`).  
**GET** this endpoint to download a ZIP with all result images.

## Requirements

- NVIDIA GPU with recent drivers
- Docker with NVIDIA Container Toolkit
- Python 3.10+ (for development outside Docker)

All dependencies are managed via Docker and `requirements.txt`.

## Development

- All main code is in `src/`
- API server: `src/api_server/cns_server.py`
- Pipelines: `src/cns/benchmark/pipeline.py`
- Utilities: `src/utils/veryutils.py`
- Requirements: `src/requirements.txt`
- Dockerfile: root of the repo

## Advanced

- **Custom checkpoints**: Place your `.pth` files in `checkpoints/`
- **Configuration**: Use JSON configs or API parameters
- **Visualization**: Output images are saved in `output_images/` and zipped for download

## Troubleshooting

Hey mate, If `torch.cuda.is_available()` is `False` do not let the fear take control over your terminal!

If you have a GPU, there are two reason for this:
  - You have installed the wrong torch version, then it's your fault... yep, sorry mate but that's true.

  - Well... the not really funny thing is that CNS's GNN needs checkpoints made with the correct version of CUDA drivers on a Nvidia GPU that use those drivers. 

In the second case, you must retrain the network with your own drivers using the 'train_cns.py' script.
Or... use the CPU, with ORB detector the system run really well on a machine with Intel i9 9900k at 5GHz and 32GB RAM.


## License

MIT License.  
See [LICENSE](LICENSE) for details.

## Authors

TeamPA-25