FROM node:22-alpine
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apk --no-cache add git bash python3 py3-pip curl
RUN npm install -g @anthropic-ai/claude-agent-sdk

RUN adduser -u 10001 -D -s /bin/bash sandboxuser
USER sandboxuser
WORKDIR /app

COPY package.json .
RUN npm install
COPY . .

ENV NODE_ENV=production

