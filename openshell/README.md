# OpenShell

A safe, private runtime for AI agents. Providing sandboxed environments from Nvidia.

"Out of process enforcement"

### Core components
Gateway - control-plane API reconciles the sandbox lifecycle
Sandbox - the isolated runtime with policy-enforced egress routing
Policy Engine - enforce constraints at filesystem, network, and process layers

### Enforcement
Filesystem - prevents reads/writes outside allowed paths
Network - blocks outbound connections
Processs - blocks privolege escaltion and "dangerous" syscalls

## Usage

Switched to using openshell by default for local agent development. It is safer...

Primary harness/agent I use is `claude code`. I sometimes use `codex` and have used a number of other similar tools. 

### Creating a policy

```yaml
version: 1

filesystem_policy:
  include_workdir: true
  read_only:
    - /usr
    - /lib
    - /proc
    - /dev/urandom
    - /app
    - /etc
    - /var/log
  read_write:
    - /sandbox
    - /tmp
    - /dev/null


network_policies:
  claude_code:
    name: openshell-claude
    endpoints:
      - { host: 192.168.4.24, port: 8889, protocol: rest, enforcement: enforce, access: full }
      - { host: github.com, port: 443 }
      - { host: release-assets.githubusercontent.com, port: 443 }
      - { host: go.dev, port: 443 }
      - { host: dl.google.com, port: 443 }
      - { host: proxy.golang.org, port: 443 }
      - { host: sum.golang.org, port: 443 }
    binaries:
      - path: /usr/local/bin/claude
      - path: /usr/bin/git
      - path: /usr/bin/node
      - path: /usr/local/bin/node
      - path: /sandbox/bin/kubebuilder
      - path: /usr/bin/curl
      - path: /sandbox/opt/go/bin/go

process:
  run_as_user: sandbox
  run_as_group: sandbox

```
### Updating Policy
There are time when the need to update the policy without recreating the sandbox.

```bash
openshell policy update \
  --add-endpoint sum.golang.org:443 \
  --binary /sandbox/go/bin/go \
  --wait
```
### Get Policy

```bash
openshell policy get --full <name>
```


### Creating a sandbox

```bash
openshell sandbox create --name demo \
    --upload ./setup-sandbox.sh:/sandbox/ \
    --policy ./policy.yaml \
    --no-auto-providers  \
    --env ANTHROPIC_AUTH_TOKEN=llama \
    --env ANTHROPIC_BASE_URL=http://192.168.4.24:8889
```

### Connect
```bash
openshell sandbox connect <sandbox-name>
```

### Monitor logs

```bash
openshell logs <sandbox-name> --tail
```

### transfer files

```bash
# upload
openshell sandbox upload <sandbox-name> /path/to/file|dir /sandbox/<target>

# download
openshell sandbox download <sandbox-name> /sandbox/path/to/file|dir
```

### custom sandbox container

- https://github.com/NVIDIA/OpenShell-Community/tree/main/sandboxes

example: Add go to the environment

```
FROM ghcr.io/nvidia/openshell-community/sandboxes/base

# Install Golang
RUN apt-get update && apt-get install -y golang-go

# Setup working directory for the sandbox user
WORKDIR /sandbox
USER 998
```
Build container to be used by openshell

```bash
docker build -t openshell-golang-go:v1 .

openshell sandbox create --name <sandbox-name> \
    --from golang-go:v1 \
    --upload ./setup-sandbox.sh:/sandbox/ \
    --policy ./policy.yaml \
    --no-auto-providers  \
    --env ANTHROPIC_AUTH_TOKEN=llama \
    --env ANTHROPIC_BASE_URL=http://192.168.4.24:8889
```

Build sandbox from a Dockerfile

```bash
openshell sandbox create \
    --name "pi" \
    --from "./Dockerfile" \
    --policy "./policy.yaml"
```

### TUI Monitor

Tool that allows to monitor state of one to many sandboxes. Review submitted rules to allow/reject endpoints an agent requests access to.

```bash
openshell term
```

# Sources
- https://github.com/NVIDIA/OpenShell
- https://github.com/NVIDIA/OpenShell-Community
- https://docs.nvidia.com/openshell/sandboxes/policies
- https://docs.nvidia.com/openshell/sandboxes/manage-sandboxes




