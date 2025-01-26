CHEKPOIT_PATH=$PWD/checkpoints
LOCAL_DATA_PATH=data

echo CHEKPOIT_PATH=$CHEKPOIT_PATH
# Download the model

docker build -t bria-4b-adapt -f Dockerfile .

#if docker container is already running, stop it
docker stop bria-train
docker rm bria-train

docker run  \
    --gpus all \
    -v $CHEKPOIT_PATH:/gradient/checkpoints \
    --name bria-train \
    -v /home/ubuntu/.aws:/root/.aws:ro \
    -v /home/ubuntu/.cache/huggingface/hub:/gradient/.cache/huggingface/hub \
    -e HF_API_TOKEN=$HF_API_TOKEN \
    bria-4b-adapt 