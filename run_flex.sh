docker run --runtime=nvidia -v $(pwd):/workspace -w /workspace -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix -p 8888:8888 -ti pyflex2 bash

