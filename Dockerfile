# OmniBioAI — AI/Bioinformatics Development Environment
# Purpose: Build the GPU AI/bioinformatics development environment.
# Author: Manish Kumar <manish@omnibioai.org>

# ==========================================================
# Image Name: omnibioai-dev-env
# Description: Full AI/Bioinformatics Development Environment
# Base: NVIDIA PyTorch 25.10 (CUDA-enabled, DGX/GPU required)
# Includes: PyTorch GPU · R/Bioconductor · JupyterLab ·
#           GATK · Samtools · FastQC · Nextflow · Ollama
# Maintainer: Manish Kumar
# GitHub: https://github.com/OmniBioAI/omnibioai-dev-docker
# Pull:  docker pull ghcr.io/omnibioai/omnibioai-dev-env:latest
# ==========================================================

FROM nvcr.io/nvidia/pytorch:25.10-py3

LABEL org.opencontainers.image.source=https://github.com/OmniBioAI/omnibioai-dev-docker
LABEL org.opencontainers.image.title="OmniBioAI Dev Environment"
LABEL org.opencontainers.image.description="Full AI/bioinformatics dev environment for DGX/GPU machines. PyTorch GPU, R/Bioconductor, GATK, Nextflow, JupyterLab, Ollama."
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Manish Kumar"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL omnibioai.type="dev-environment"
LABEL omnibioai.requires-gpu="true"
LABEL omnibioai.requires-nvidia-container-toolkit="true"
LABEL omnibioai.platform="OmniBioAI"

WORKDIR /workspace

# ==========================================================
# 1. System dependencies
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libmysqlclient-dev \
    pkg-config \
    software-properties-common \
    dirmngr \
    gnupg \
    apt-transport-https \
    ca-certificates \
    zstd \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 2. R + Bioconductor
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/* \
    && R -e "install.packages(c('tidyverse', 'data.table', 'BiocManager', 'devtools'), repos='https://cloud.r-project.org')" \
    && R -e "BiocManager::install(c('ComplexHeatmap', 'limma', 'edgeR', 'DESeq2', 'SingleCellExperiment', 'scran', 'scater'), ask=FALSE)"

# ==========================================================
# 3. Python stack
# Note: torch, torchvision, flash_attn, apex, transformer_engine
#       etc. are pre-installed in the nvcr.io base — not reinstalled
# ==========================================================
COPY requirements.txt /tmp/requirements.txt
COPY requirements.dgx.txt /tmp/requirements.dgx.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ==========================================================
# 4. Ollama
# ==========================================================
RUN curl -fsSL https://ollama.com/install.sh | bash \
    && ollama --version

# ==========================================================
# 5. JupyterLab configuration
# ==========================================================
RUN mkdir -p /root/.jupyter \
    && printf "%s\n" \
        "c.NotebookApp.ip = '0.0.0.0'" \
        "c.NotebookApp.open_browser = False" \
        "c.NotebookApp.allow_root = True" \
        "c.NotebookApp.token = ''" \
        "c.NotebookApp.password = ''" \
        "c.ServerApp.ip = '0.0.0.0'" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.allow_root = True" \
        >> /root/.jupyter/jupyter_notebook_config.py

# ==========================================================
# 6. Bioinformatics tools — Java + system tools
# ==========================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    default-jdk \
    fastqc \
    samtools \
    bcftools \
    bedtools \
    tabix \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# 7. Nextflow
# ==========================================================
RUN curl -fsSL https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin/ \
    && chmod +x /usr/local/bin/nextflow \
    && nextflow -version

# ==========================================================
# 8. GATK 4.5.0.0
# ==========================================================
RUN curl -fsSL -o /tmp/gatk-4.5.0.0.zip     https://github.com/broadinstitute/gatk/releases/download/4.5.0.0/gatk-4.5.0.0.zip     && unzip /tmp/gatk-4.5.0.0.zip -d /opt     && ln -s /opt/gatk-4.5.0.0/gatk /usr/local/bin/gatk     && rm /tmp/gatk-4.5.0.0.zip     && gatk --version

# ==========================================================
# 9. SnpEff
# ==========================================================
RUN curl -fsSL -o /tmp/snpEff_latest_core.zip \
    https://sourceforge.net/projects/snpeff/files/snpEff_latest_core.zip/download \
    && unzip /tmp/snpEff_latest_core.zip -d /opt \
    && ln -s /opt/snpEff/snpEff.jar /usr/local/bin/snpEff.jar \
    && printf '#!/bin/bash\njava -jar /usr/local/bin/snpEff.jar "$@"\n' \
        > /usr/local/bin/snpeff \
    && chmod +x /usr/local/bin/snpeff \
    && rm /tmp/snpEff_latest_core.zip

# ==========================================================
# 10. Environment
# ==========================================================
ENV PATH="/usr/local/bin:/opt/gatk-4.5.0.0:/opt/snpEff:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/root/.cache/huggingface
ENV OLLAMA_HOST=0.0.0.0
ENV JUPYTER_PORT=8888
ENV OLLAMA_PORT=11434

# ==========================================================
# 11. Validation — confirm key tools are installed
# ==========================================================
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')" \
    && python -c "import transformers; print(f'Transformers {transformers.__version__}')" \
    && nextflow -version \
    && gatk --version \
    && samtools --version | head -1 \
    && fastqc --version \
    && R --version | head -1 \
    && echo "✅ All tools validated"

# ==========================================================
# 12. Ports
# 8888 → JupyterLab  |  11434 → Ollama
# ==========================================================
EXPOSE 8888
EXPOSE 11434

# Entrypoint and default command
CMD ["bash"]
