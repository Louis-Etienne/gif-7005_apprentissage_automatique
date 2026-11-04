docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -v ./data:/workspace/data \
  -v ./checkpoints:/workspace/checkpoints \
  -v ./outputs:/workspace/outputs \
  mammo-fm:cuda11.8

nvidia-smi