FROM python:3.10

WORKDIR /outlines

RUN pip install --upgrade pip

# Copy necessary build components
COPY .git ./.git
COPY pyproject.toml .
COPY outlines ./outlines

# Install outlines and outlines[serve]
RUN pip install .[serve]

# https://outlines-dev.github.io/outlines/reference/vllm/
ENTRYPOINT python3 -m outlines.serve.serve
