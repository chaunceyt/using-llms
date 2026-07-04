#!/bin/bash

mkdir $HOME/opt
mkdir -p $HOME/workspace/go


# Setup go
curl -s -LO "https://go.dev/dl/go1.26.4.linux-arm64.tar.gz"
tar -C $HOME/opt -xzf go*.linux-arm64.tar.gz
echo 'export PATH=$PATH:$HOME/opt/go/bin' >> /sandbox/.bashrc
rm  go*.linux-arm64.tar.gz
echo 'export GOROOT="$HOME/opt/go"' >> /sandbox/.bashrc
echo 'export GOPATH="$HOME/workspace/go"' >> /sandbox/.bashrc

# Setup kubebuilder
mkdir /sandbox/bin
curl -s -Lo /sandbox/bin/kubebuilder "https://github.com/kubernetes-sigs/kubebuilder/releases/download/v4.15.0/kubebuilder_linux_arm64"
chmod +x /sandbox/bin/kubebuilder
echo 'export PATH=$PATH:/sandbox/bin' >> /sandbox/.bashrc

git clone https://github.com/kedacore/keda.git

source .bashrc
