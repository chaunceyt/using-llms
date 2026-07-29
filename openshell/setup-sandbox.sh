#!/bin/bash

mkdir $HOME/opt
mkdir -p $HOME/workspace/go
mkdir $HOME/agent
mkdir -p $HOME/.claude


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

echo 'export ANTHROPIC_AUTH_TOKEN=llama' >> /sandbox/.bashrc
echo 'export ANTHROPIC_BASE_URL=http://192.168.4.24:8899' >> /sandbox/.bashrc

# Install claude-agent-sdk
npm install @anthropic-ai/claude-agent-sdk
npm install -D typescript @types/node tsx

if [ ! -f /sandbox/.claude.json ]; then
    printf '{"trustedFolders":["/sandbox","/sandbox/agent"],"hasCompletedOnboarding":true,"projects":{"/sandbox":{"hasTrustDialogAccepted":true},"/sandbox/agent":{"hasTrustDialogAccepted":true}}}\n' > /sandbox/.claude.json
fi
if [ ! -f /sandbox/.claude/settings.json ]; then
    printf '{"theme":"dark"}\n' > /sandbox/.claude/settings.json
fi

git clone https://github.com/chaunceyt/aichat-workspace-operator.git
source .bashrc
