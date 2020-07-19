FROM nvidia/cudagl:10.2-devel-ubuntu18.04
RUN apt update && apt install -y freeglut3-dev build-essential python3-pip \
  python3
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN mkdir /workspace
WORKDIR /workspace
COPY requirements-gpu.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html

