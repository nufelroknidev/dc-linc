# ── BUILD STAGE: compile Prover9 & Mace4 ─────────────────────────────────────
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl build-essential autoconf bzip2 ca-certificates make tar && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

# download & extract LADR‑2009‑11A
RUN curl -fsSL https://www.cs.unm.edu/~mccune/prover9/download/LADR-2009-11A.tar.gz \
    | tar xz

# patch Makefile so -lm is at the end of link lines
RUN sed -i 's/-lm -o/ -o/g' LADR-2009-11A/provers.src/Makefile \
    && sed -i '/libladr\.a/ s/$/ -lm/' LADR-2009-11A/provers.src/Makefile

WORKDIR /usr/src/LADR-2009-11A
RUN make all


# ── RUNTIME STAGE: CUDA runtime + Python + tools ─────────────────────────────
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

# install python3.10 & minimal runtime deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-utils \
      git \
      python3.10 \
      python3.10-venv \
      python3.10-distutils \
      python3-pip \
      libgmp10 \
      make \
      curl && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
    rm -rf /var/lib/apt/lists/*

# copy in prover9 & mace4 from builder
COPY --from=builder /usr/src/LADR-2009-11A/bin/prover9 /usr/local/bin/
COPY --from=builder /usr/src/LADR-2009-11A/bin/mace4   /usr/local/bin/
RUN chmod +x /usr/local/bin/prover9 /usr/local/bin/mace4

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p /opt/conda && \
    rm /tmp/conda.sh && \
    /opt/conda/bin/conda clean -afy && \
    ln -s /opt/conda/bin/conda /usr/local/bin/conda

# Use bash login shell so 'conda activate' works
SHELL ["bash", "-lc"]

# Accept Anaconda Terms of Service and create environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set always_yes yes --set changeps1 no && \
    conda create -n linc python=3.10 && \
    conda clean -afy

# Update PATH to include conda environment
ENV PATH="/opt/conda/envs/linc/bin:/opt/conda/bin:${PATH}"

# clone your LINC repo
RUN git clone https://github.com/NufelRokni/linc.git /app/linc

WORKDIR /app/linc

# default to an interactive shell (or override with your entrypoint)
CMD ["bash"]
