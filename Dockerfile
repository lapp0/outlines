FROM python:3.10

WORKDIR /outlines

RUN pip install --upgrade pip

# Install outlines and outlines[serve]
RUN pip install .
RUN pip install .[serve]

# https://outlines-dev.github.io/outlines/reference/vllm/
ENTRYPOINT python3 -m outlines.serve.serve
