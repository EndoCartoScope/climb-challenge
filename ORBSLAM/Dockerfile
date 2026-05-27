FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Build context expected: ORBSLAM/  (i.e. this directory).
# Contains:
#   - ORB-SLAM3/   (vendored source tree)
#   - entrypoint.sh (orchestrator that runs all sequences x runs)
#   - Dockerfile

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=offscreen \
    PANGOLIN_WINDOW_URI=headless:// \
    ORB_SLAM3_ROOT=/opt/ORB_SLAM3 \
    LD_LIBRARY_PATH=/opt/ORB_SLAM3/lib:/opt/ORB_SLAM3/Thirdparty/DBoW2/lib:/opt/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config curl ca-certificates unzip \
    libopencv-dev \
    libeigen3-dev \
    libatlas-base-dev liblapack-dev \
    libsuitesparse-dev \
    libglew-dev libgl1-mesa-dev libepoxy-dev xorg-dev \
    libgtk2.0-dev libjpeg-dev libpng-dev libtiff-dev \
    libboost-dev libboost-system-dev libboost-filesystem-dev libboost-serialization-dev \
    libssl-dev \
    ffmpeg python3 python3-pip \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Pangolin (required by ORB-SLAM3 viewer)
RUN git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin && \
    cmake -S /opt/Pangolin -B /opt/Pangolin/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PANGOLIN_FFMPEG=OFF \
      -DBUILD_PANGOLIN_PYTHON=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF && \
    cmake --build /opt/Pangolin/build -j && \
    cmake --install /opt/Pangolin/build && \
    rm -rf /opt/Pangolin

RUN python3 -m pip install --upgrade pip && \
    pip install numpy

COPY ORB-SLAM3 /opt/ORB_SLAM3
WORKDIR /opt/ORB_SLAM3

RUN rm -rf \
      build \
      Thirdparty/DBoW2/build \
      Thirdparty/g2o/build \
      Thirdparty/Sophus/build && \
    sed -i 's/ORB_SLAM3::System SLAM(argv\[1\],argv\[2\],ORB_SLAM3::System::MONOCULAR,true)/ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,false)/' \
      Examples/Monocular/mono_endo_hculb.cc && \
    cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Thirdparty/DBoW2/build -j && \
    cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Thirdparty/g2o/build -j && \
    cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build Thirdparty/Sophus/build -j && \
    if [ ! -f Vocabulary/ORBvoc.txt ]; then tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary; fi && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j4 && \
    test -s lib/libORB_SLAM3.so && \
    test -s Vocabulary/ORBvoc.txt

# Allow any runtime --user to write the per-run temp output dir under cwd.
RUN chmod -R a+rwX /opt/ORB_SLAM3

COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/input", "/output"]
