FROM dustynv/l4t-pytorch:r32.7.1

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH
RUN pip3 install requests matplotlib scipy

ENV HOME=/home
WORKDIR /home
RUN git clone https://github.com/haotruongnhat/battery

ENTRYPOINT ["tail", "-f", "/dev/null"]
CMD ["/bin/bash"]