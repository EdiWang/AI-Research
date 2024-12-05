# Phi-3 with ONNX

## Build Backend

```bash
cd backend

sudo apt install pipx
pipx install huggingface-hub
pipx ensurepath
# logout and login again

huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

docker build -t phi3-onnx-backend .
```

## Run Backend

```bash
docker run -p 8000:8000 phi3-onnx-backend
```

