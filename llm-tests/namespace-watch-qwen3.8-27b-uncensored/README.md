# namespace-watcher

A Kubernetes **namespace watcher** written in Go. When a namespace is created,
it provisions a set of "sane default" resources into that namespace so every
new namespace starts out consistently configured.

## What it creates

For each new namespace the watcher creates the following resources (all built
with the [unstructured](https://pkg.go.dev/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured)
package):

| Resource       | Name (default)         | Purpose                                                    |
| -------------- | ---------------------- | ---------------------------------------------------------- |
| `LimitRange`   | `default-limits`       | Per-container default / default-request / max CPU & memory |
| `ResourceQuota`| `production-quota`     | Aggregate namespace limits (CPU, memory, pods, etc.)       |
| `ServiceAccount` | `myapp-sa`           | Service account for the app                                |
| `Role`         | `myapp-role`           | Read-only (`get`, `list`) access to configmaps             |
| `RoleBinding`  | `myapp-rolebinding`    | Binds the role to the service account                      |
| `NetworkPolicy`| `myapp-network-policy` | Ingress only from the ingress controller; egress only to postgres + DNS |

Every created resource is labelled with `app.kubernetes.io/managed-by=namespace-watcher`
so the resources it owns are easy to discover and reconcile.

All of the concrete values (names, quantities, labels, ports, etc.) live in a
single `config.Defaults` struct — see [Configuration](#configuration) — so they
can be changed in one place without touching the provisioning logic.

## Behavior

- **New-only (default):** only namespaces created *after* the watcher starts are
  provisioned. Pre-existing namespaces are left alone.
- **Backfill:** pass `--backfill` to also provision namespaces that already
  exist when the watcher starts.
- **Idempotent:** each namespace is provisioned at most once, and any resource
  that already exists is skipped. Safe to run multiple instances or restarts.

## Architecture

The code is split into small, focused packages:

```
.
├── main.go                     # flags, config wiring, graceful shutdown
├── internal/
│   ├── config/                 # the "sane defaults" (all tunable values)
│   │   └── config.go
│   ├── provisioner/            # builds + creates the resources (unstructured)
│   │   ├── provisioner.go      # idempotent create via the dynamic client
│   │   ├── templates.go        # one builder per resource type
│   │   ├── mapper.go           # static RESTMapper (no runtime discovery)
│   │   └── *_test.go
│   └── watcher/                # namespace informer + bounded work queue
│       ├── watcher.go
│       └── watcher_test.go
├── deploy/
│   └── namespace-watcher.yaml  # Namespace/SA/ClusterRole/Binding/Deployment
├── Dockerfile
└── Makefile
```

Key design decisions:

- **client-go informer** on `v1/namespaces` for watching. Events are fanned out
  to a **bounded work queue** processed by a small pool of workers, so a burst
  of namespace creations never blocks the informer and work is deduplicated per
  namespace.
- **`unstructured` + dynamic client** for all resource creation. The set of
  resources and their values are data-driven from `config.Defaults`, so changing
  the defaults does not require touching the create path.
- **Static RESTMapper** instead of a runtime discovery call. The watcher creates a
  fixed, known set of built-in resources, so a static `meta.RESTMapper` is
  deterministic, needs no startup discovery call, and keeps the RBAC surface
  minimal.
- **Least-privilege RBAC:** the watcher only needs to watch namespaces and
  create/read the six resource types it provisions.

## Requirements

- Go 1.22+
- Access to a Kubernetes cluster (in-cluster service account, or a kubeconfig)

## Build

```sh
make build          # -> bin/namespace-watcher
```

## Run

Run against the in-cluster service account:

```sh
./bin/namespace-watcher
```

Run locally against a kubeconfig:

```sh
./bin/namespace-watcher --kubeconfig=$HOME/.kube/config
```

Backfill existing namespaces:

```sh
./bin/namespace-watcher --backfill
```

### Flags

| Flag            | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `--kubeconfig`  | `""`    | Path to a kubeconfig. Empty = in-cluster, else default kubeconfig location |
| `--backfill`    | `false` | Also provision namespaces that already exist on startup |
| `--concurrency` | `4`     | Number of workers provisioning namespaces               |
| `--log-level`   | `info`  | `debug`, `info`, `warn`, or `error`                     |

## Configuration

The default values are defined in
[`internal/config/config.go`](internal/config/config.go). `config.NewDefaultConfig()`
returns a fully populated `config.Defaults`. To change any default, edit that
function. To change the values at runtime, construct your own `config.Defaults`
and pass it to `provisioner.New` / `provisioner.NewStaticMapper` in `main.go`.

## Deploy to a cluster

```sh
# Build and push an image (point the Dockerfile output at your registry).
docker build -t <registry>/namespace-watcher:latest .
docker push <registry>/namespace-watcher:latest
```

Update the `image:` field in [`deploy/namespace-watcher.yaml`](deploy/namespace-watcher.yaml)
to your image, then:

```sh
kubectl apply -f deploy/namespace-watcher.yaml
```

Create a namespace and watch it get provisioned:

```sh
kubectl create namespace demo
kubectl -n demo get limitrange,resourcequota,serviceaccount,role,rolebinding,networkpolicy
```

## Test

```sh
make test           # go test ./...
make vet            # go vet ./...
```

The test suite covers:

- Each unstructured template (GVK, name, namespace, labels, and the key spec
  fields, verified by round-tripping to the typed API object).
- The provisioner end-to-end against a fake dynamic client (all six resources
  are created; idempotency on re-runs; `MarkDone` skips provisioning).
- The static RESTMapper (every GVK resolves to the correct REST resource).
- The watcher's new-only vs. backfill behavior using a fake clientset and a real
  informer.
