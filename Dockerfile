ARG CUDA_VERSION=12.8.1
FROM docker.1ms.run/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0
ARG CPU_VARIANT=x86-intel-multi
ARG BUILD_ALL_CPU_VARIANTS=1

# Proxy settings for build-time network access
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG HOPPER_SBO_DEEPEP_COMMIT=9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG SGL_VERSION=0.5.6.post1
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG UBUNTU_MIRROR=0
ARG GITHUB_ARTIFACTORY=github.com
ARG FLASHINFER_VERSION=0.5.3

# ktransformers wheel version (cu128torch28 for CUDA 12.8 + PyTorch 2.8)
ARG KTRANSFORMERS_VERSION=0.5.3
ARG KTRANSFORMERS_WHEEL=ktransformers-0.4.2+cu128torch28fancy-cp312-cp312-linux_x86_64.whl

# flash_attn wheel for fine-tune env
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="/opt/miniconda3/bin:${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Core system utilities
    tzdata \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    # Development tools
    gdb \
    ninja-build \
    rdma-core \
    # NCCL
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \

    # Install conda
    && mkdir -p /opt/miniconda3 \
    && wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh \
    && bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 \
    && rm /opt/miniconda3/miniconda.sh && /opt/miniconda3/bin/conda clean --all \

    # Accept conda TOS
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \

    # Configure conda to use Tsinghua mirror
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free \
    && conda config --set show_channel_urls yes

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8


########################################################
########## Dual Conda Environment Setup ################
########################################################

FROM base AS framework

ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG SGL_KERNEL_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG FLASHINFER_VERSION
ARG GRACE_BLACKWELL
ARG GRACE_BLACKWELL_DEEPEP_BRANCH
ARG HOPPER_SBO
ARG HOPPER_SBO_DEEPEP_COMMIT
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY
ARG KTRANSFORMERS_VERSION
ARG KTRANSFORMERS_WHEEL
ARG FLASH_ATTN_WHEEL
ARG FUNCTIONALITY=0

WORKDIR /workspace

# Create conda environments (fine-tune only needed for sft mode)
RUN conda create -n serve python=3.12 -y && conda clean --all \

  # Initialize conda for bash
  && /opt/miniconda3/bin/conda init bash \

  # Create shell aliases for convenience
  && echo '\n# Conda environment aliases\nalias serve="conda activate serve"' >> /root/.bashrc \

  # Set pip mirror for conda envs
  && /opt/miniconda3/envs/serve/bin/pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Upgrade pip and install basic tools in serve env
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate serve && \
    
    pip install --upgrade pip setuptools wheel html5lib six uv && \
    
    # Install sgl-kernel
    pip install sgl-kernel==${SGL_KERNEL_VERSION}
    
# Clone repositories (sglang is included as a submodule in ktransformers)
RUN git clone --depth 1 https://${GITHUB_ARTIFACTORY}/kvcache-ai/ktransformers.git /workspace/ktransformers \
    && cd /workspace/ktransformers \
    && git submodule update --init --recursive \
    && ln -s /workspace/ktransformers/third_party/sglang /workspace/sglang
    
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . /opt/miniconda3/etc/profile.d/conda.sh \
    && conda activate serve \
    
    # Install sglang-kt(version aligned with ktransformers)
    && export SGLANG_KT_VERSION=$(python3 -c "exec(open('/workspace/ktransformers/version.py').read()); print(__version__)") \
    && echo "Installing sglang-kt v${SGLANG_KT_VERSION}" \
    && cd /workspace/sglang \
    && uv pip install -e "python[all]" --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/cu${CUINDEX}  --index-strategy unsafe-best-match
    
RUN FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning \
    /opt/miniconda3/envs/serve/bin/python -m flashinfer --download-cubin
    
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    . /opt/miniconda3/etc/profile.d/conda.sh \
    && conda activate serve \
    # Install NCCL for serve env
    && pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps \

    # Install kt-kernel with avx2
    && cd /workspace/ktransformers/kt-kernel \
    && sed -i.bak 's/python3 -m pip/uv pip/g' install.sh \
    && CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_ENABLE_AMX=OFF ./install.sh build --manual \
    
    # fix nvidia-cudnn-cu12
    && uv pip install --force-reinstall nvidia-cudnn-cu12==9.16.0.29

# Extract versions from each component and save to versions.env
RUN set -x && \
    # KTransformers version (single source of truth for both kt-kernel and sglang-kt)
    cd /workspace/ktransformers && \
    KTRANSFORMERS_VERSION=$(python3 -c "exec(open('version.py').read()); print(__version__)" 2>/dev/null || echo "unknown") && \
    echo "KTRANSFORMERS_VERSION=$KTRANSFORMERS_VERSION" > /workspace/versions.env && \
    echo "Extracted KTransformers version: $KTRANSFORMERS_VERSION" && \
    \
    # sglang-kt version = ktransformers version (aligned)
    echo "SGLANG_KT_VERSION=$KTRANSFORMERS_VERSION" >> /workspace/versions.env && \
    echo "sglang-kt version (aligned): $KTRANSFORMERS_VERSION" && \
    \
    # LLaMA-Factory version (from fine-tune environment, sft mode only)
    if [ "$FUNCTIONALITY" = "sft" ]; then \
        . /opt/miniconda3/etc/profile.d/conda.sh && conda activate fine-tune && \
        cd /workspace/LLaMA-Factory && \
        LLAMAFACTORY_VERSION=$(python -c "import sys; sys.path.insert(0, 'src'); from llamafactory import __version__; print(__version__)" 2>/dev/null || echo "unknown") && \
        echo "LLAMAFACTORY_VERSION=$LLAMAFACTORY_VERSION" >> /workspace/versions.env && \
        echo "Extracted LLaMA-Factory version: $LLAMAFACTORY_VERSION"; \
    else \
        echo "LLAMAFACTORY_VERSION=none" >> /workspace/versions.env && \
        echo "LLaMA-Factory not installed (infer mode)"; \
    fi && \
    \
    # Display all versions
    echo "=== Version Summary ===" && \
    cat /workspace/versions.env

CMD ["/bin/bash"]
