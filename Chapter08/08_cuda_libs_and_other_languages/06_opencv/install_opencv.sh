#!/bin/bash
# Install dependencies
sudo apt-get install -y --no-install-recommends \
    cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libatlas-base-dev gfortran libeigen3-dev \
    libgtkglext1 libgtkglext1-dev

# We are install OpenCV 4.1.1
OPENCV_VERSION=4.1.1
OPENCV_DIR=opencv

# Download OpenCV and contrib source codes
mkdir -p ${OPENCV_DIR}
wget -O ${OPENCV_DIR}/opencv-${OPENCV_VERSION}.tar.gz https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz
wget -O ${OPENCV_DIR}/opencv_contrib-${OPENCV_VERSION}.tar.gz https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz

# Untar the files
tar -C ${OPENCV_DIR} -xzf ${OPENCV_DIR}/opencv-${OPENCV_VERSION}.tar.gz
tar -C ${OPENCV_DIR} -xzf ${OPENCV_DIR}/opencv_contrib-${OPENCV_VERSION}.tar.gz

# Build the codes & install
cd ${OPENCV_DIR}/opencv-${OPENCV_VERSION}
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_CUDA=ON -D WITH_CUVID=ON -D BUILD_opencv_cudacodec=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
    -D WITH_CUBLAS=1 ..
    -D PYTHON_DEFAULT_EXECUTABLE=`which python3` \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_EXAMPLES=ON ..
make -j$(nproc)
sudo make install -j$(nproc)