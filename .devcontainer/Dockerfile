FROM debian:bookworm-slim

# Install essential packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    g++ \
    make \
    libopenmpi-dev \
    openmpi-bin \
    openssh-client \
    git \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Build Boost 1.89 with MPI and serialization
RUN mkdir /boost && \
    cd /boost && \
    wget "https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz" && \
    tar xzf boost_1_82_0.tar.gz && \
    cd boost_1_82_0 && \
    ./bootstrap.sh --with-libraries=mpi,serialization && \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 && \
    rm -f /boost/boost_1_82_0.tar.gz

# Create non-root user
RUN addgroup --gid 1000 mpiuser && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" mpiuser

USER mpiuser

# Environment variables for Boost
ENV BOOST_CFLAGS=-I/boost/boost_1_82_0/
ENV BOOST_LIBS=-L/boost/boost_1_82_0/stage/lib
ENV LD_LIBRARY_PATH=/boost/boost_1_82_0/stage/lib:${LD_LIBRARY_PATH}