FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/submission
COPY run.py /opt/submission/run.py

ENTRYPOINT ["python3", "/opt/submission/run.py"]
CMD ["--input", "/input", "--output", "/output"]
