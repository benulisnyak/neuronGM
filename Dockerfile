FROM ubuntu:xenial

# Set the working directory to /app
WORKDIR /app

# Install build system and Boost library
RUN apt-get update \
    && apt-get install --yes wget build-essential gcc-multilib libboost-all-dev

# Install GSL
RUN wget -O gsl.tgz ftp://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz \
    && tar -zxf gsl.tgz \
    && mkdir gsl \
    && cd gsl-1.16 \
    && ./configure  \
    && make \
    && make install

# Install SimKernel
RUN apt-get install --yes unzip \
    && wget -O simkernel.zip http://github.com/ChristophKirst/SimKernel/archive/master.zip \
    && unzip simkernel.zip \
    && cd SimKernel-master \
    && make \
    && make install

# Install Eigen
RUN wget -O eigen.tgz https://gitlab.com/libeigen/eigen/-/archive/3.2.7/eigen-3.2.7.tar.gz \
    && tar -zxf eigen.tgz -C /app \
    && cp -R eigen-3.2.7/ /usr/local/include/eigen3/

RUN apt-get install --yes autotools-dev autoconf

COPY . /app

RUN aclocal \
    && autoconf \
    && autoheader \
    && automake --add-missing \
    && ./configure

