ARG PYTORCH_VERSION=2.7.0
ARG CUDA_VERSION=12.6
ARG CUDNN_VERSION=9

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime
LABEL maintainer=jhmipc@jh.edu
ARG PYTORCH_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION

ENV PYTHONUSERBASE=/opt/conda

RUN echo -e "{\n \
    \"PYTORCH_VERSION\": \"${PYTORCH_VERSION}\",\n \
    \"CUDA_VERSION\": \"${CUDA_VERSION}\",\n \
    \"CUDNN_VERSION\": \"${CUDNN_VERSION}\"\n \
}" > /opt/manifest.json

RUN apt-get update && \
    apt-get -y --no-install-recommends install ca-certificates git

COPY requirements.txt /tmp/ddpm3d-mri/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/ddpm3d-mri/requirements.txt && \
    rm -rf /tmp/ddpm3d-mri

COPY . /tmp/ddpm3d-mri

RUN pip install --no-deps --no-cache-dir /tmp/ddpm3d-mri && \
    rm -rf /tmp/ddpm3d-mri

ENTRYPOINT ["python"]