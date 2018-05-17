[![Build Status](https://travis-ci.com/poanetwork/hbbft.svg?branch=master)](https://travis-ci.com/poanetwork/hbbft) 
[![Gitter](https://badges.gitter.im/poanetwork/hbbft.svg)](https://gitter.im/poanetwork/hbbft?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# About

An implementation of the paper
["Honey Badger of BFT Protocols"](https://eprint.iacr.org/2016/199.pdf)
in Rust. This is a modular library of consensus. There are
[examples](./examples/README.md) illustrating the use of this algorithm.

**This is work in progress.** Parts of the algorithm are still missing
or incomplete.

An example is included to run a simulation of a network:

    $ cargo run --example simulation -- --nodes 10 --faulty 1 --txs 1000 --batch 100

# Requirements

`hbbft` has optional protobuf support. To use protobuf, enable the feature
`serialization-protobuf` in your `Cargo.toml`, and make sure you have
Google's Protocol Buffer Compiler, `protoc` binary, located somewhere in
your `$PATH`. You must be using Protocol Buffer Compiler version 3 or greater.
Running any of the following install methods will save a `protoc` binary at
`/usr/local/bin/protoc`.

*Note:* as of writing this, the latest stable release of `protoc` is
v3.5.1. You can find out what is the latest compiler version is
[here](https://github.com/google/protobuf/releases), if you are not
installing `protoc` on Debian 9 or Ubuntu 17, change your cURL URL and zip
file names accordingly. 

## Installing `protoc` on Debian 9 (Strech) and Ubuntu 17 (Artful)

    $ sudo apt-get update
    $ sudo apt-get install -y protobuf-compiler

## Installing `protoc` on other versions of Debian and Ubuntu

    $ sudo apt-get update
    $ sudo apt-get install -y unzip
    $ cd <some temporary working directory>
    $ curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
    $ sudo unzip protoc-3.5.1-linux-x86_64.zip -d /usr/local bin/protoc
    $ sudo chown $(whoami) /usr/local/bin/protoc
    $ rm protoc-3.5.1-linux-x86_64.zip

## Installing `protoc` on OSX

    $ cd <some temporary working directory>
    $ curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-osx-x86_64.zip
    $ sudo unzip protoc-3.5.1-osx-x86_64.zip -d /usr/local bin/protoc
    $ rm protoc-3.5.1-osx-x86_64.zip

# Building

Once you have verified that the `protoc` binary is in your `$PATH`, you can
build `hbbft` using cargo:

    $ cargo build [--release]
