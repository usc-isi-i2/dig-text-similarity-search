FROM ubuntu:16.04

WORKDIR /app

## install required command utils
RUN apt-get update && apt-get install -y \
     wget \
    python3 \
    python3-pip \
    python3-venv \
    git \
    vim \
    curl \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py

# install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -p /app/miniconda -b && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/app/miniconda/bin:${PATH}
RUN conda update -y conda


# create and config conda-env
RUN cd /app \
    && git clone https://github.com/usc-isi-i2/dig-text-similarity-search \
    && cd dig-text-similarity-search \
    && conda-env create .

ENV PATH /app/miniconda/envs/dig_text_similarity/bin:$PATH
RUN conda install -c pytorch faiss-cpu


EXPOSE 5555
ENTRYPOINT ["/app/dig-text-similarity-search/docker-entrypoint.sh"]
