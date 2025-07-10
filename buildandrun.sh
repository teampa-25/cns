docker build . -t cns_sim_container

docker run -it -d \
    --name cns_sim_container \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$(pwd):/workspace" \
    -v "$HOME/.tmp:/tmp-video/" \
    -w /workspace \
    --runtime=nvidia \
    --gpus "device=1" \
    --privileged \
    cns_sim_container

#  --mount src="$(pwd)",target=/workspace/,type=bind \
# -v $(pwd):/workspace \
# -v $(pwd)/checkpoints/:/root/checkpoints/ \

docker exec -it cns_sim_container bash
clear
read -p "Do you want to stop container? (y/N):" answer

if [[ "$answer" == [Yy] ]]; then
    echo "Stopping"
    docker stop cns_sim_container
    echo "Container stopped."
else
    echo "Container will continue running."
fi
