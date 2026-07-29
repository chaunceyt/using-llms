# syntax=docker/dockerfile:1.4

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pi Coding Agent sandbox image for OpenShell
#
# Builds on the community base sandbox and adds pi.
# Build:  docker build -t openshell-pi --build-arg BASE_IMAGE=openshell-base .
# Run:    openshell sandbox create --from pi

ARG BASE_IMAGE=ghcr.io/nvidia/openshell-community/sandboxes/base:latest
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y golang-go strace unzip sqlite3 \
    && apt autoremove && apt autoremove && rm -rf /var/lib/apt/lists/*

    # Install pi
RUN npm install -g @earendil-works/pi-coding-agent
RUN npm install -g @qwen-code/qwen-code@latest

# Copy sandbox policy
COPY policy.yaml /etc/openshell/policy.yaml

USER sandbox

ENTRYPOINT ["/bin/bash"]
