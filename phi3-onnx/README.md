# Phi-3 with ONNX

POC for running Phi-3.5 model with ONNX in a docker container.

## Build Backend

```bash
cd backend

sudo apt install pipx
pipx install huggingface-hub
pipx ensurepath
# logout and login again

huggingface-cli download microsoft/Phi-3.5-mini-instruct-onnx --include cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/* --local-dir .

docker build -t phi-35-onnx-backend .
```

## Build Frontend

Please change the `API_URL` in `frontend/app.py` to the correct backend URL.

```bash
cd frontend
docker build -t phi-35-onnx-frontend .
```

## Run in the same network

Change the `API_URL` in `frontend/app.py` to `http://phi-35-onnx-backend:8000/predict`.

```bash
docker network create phi3-onnx
docker run -d --network phi3-onnx --name phi-35-onnx-backend -p 8000:8000 phi-35-onnx-backend
docker run -d --network phi3-onnx --name phi-35-onnx-frontend -p 5000:5000 phi-35-onnx-frontend
```

[Reference](https://azure.github.io/AppService/2024/08/19/Phi-3-ONNX.html)