# syntax=docker/dockerfile:1
# The P&ID model APIs (detect = RF-DETR, ocr = RapidOCR) as one arm64 Lambda image; the function's command picks the
# handler (handler.detect or handler.ocr). Build from the repo root, passing the folder that holds rfdetr_pid.pth
# (the repo only has its Git LFS pointer):
#   docker buildx build --platform linux/arm64 -f deploy/Dockerfile --build-context weights=<folder> \
#     --build-arg COMMIT=$(git rev-parse --short HEAD) -t pid-models --load .
# AWS's Python 3.12 Lambda image (Amazon Linux 2023, glibc 2.34), pinned by digest. Run outside Lambda, its entrypoint
# starts the built-in Runtime Interface Emulator on port 8080.
FROM public.ecr.aws/lambda/python:3.12@sha256:f229f304f82f5016fe26e4714d05b6a8196e0c6c61b463112841a5e3c3647163

# opencv-python needs libGL
RUN dnf install -y mesa-libGL && dnf clean all

COPY requirements.txt /tmp/requirements.txt
# torch from the CPU-only index: the PyPI build would pull GPU libraries the Lambda can never use
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

ARG WEIGHTS_SHA256=063a43b7fbc4066fa6a7ab0a999c0964839b3a40a02a17e33473c5f6502d4df6
COPY --from=weights rfdetr_pid.pth /opt/models/rfdetr_pid.pth
RUN echo "$WEIGHTS_SHA256  /opt/models/rfdetr_pid.pth" | sha256sum -c -

COPY batch/tag_pipeline.py deploy/handler.py ${LAMBDA_TASK_ROOT}/

# every model file is fetched now, into the image; at run time the filesystem is read-only except /tmp
ENV HF_HOME=/opt/cache/hf TORCH_HOME=/opt/cache/torch XDG_CACHE_HOME=/opt/cache MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=${LAMBDA_TASK_ROOT}
RUN cd /tmp && python -c "import handler; handler.warm()" && chmod -R a+rX /opt /var/lang/lib/python3.12/site-packages

ARG COMMIT=unknown
ENV PID_BUILD_COMMIT=$COMMIT PID_WEIGHTS_SHA256=$WEIGHTS_SHA256 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["handler.detect"]
