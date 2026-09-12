# Security Review: namespace-watch-qwen3.8-fn

## Scope

Full-directory scan of the namespace-watch Kubernetes controller written in Go: CLI bootstrap and credential resolution, the namespace watch/reconcile loop, the hardcoded default-resource templates, the deployment manifest with its cluster-wide RBAC, the container build, and the Go dependency manifests.

- Scan mode: repository
- Target kind: directory_snapshot
- Target ID: dir-namespace-watch-qwen3.8-fn
- Snapshot digest: sha256:1ce0b3d68d299cf157b9fabd471631334e71f123987c5d20c497682f55bf185f
- Inventory strategy: directory
- Included paths: .
- Excluded paths: none
- Runtime or test status: Static source analysis only; the controller was not built or executed.
- Artifacts reviewed: Dockerfile, cmd/namespace-watch/main.go, deploy/deploy.yaml, go.mod, go.sum, internal/controller/controller.go, internal/controller/controller_test.go, internal/defaults/defaults.go, internal/defaults/defaults_test.go

Limitations and exclusions:
- The scan root is not a git repository; only the current directory state was reviewed, with no revision history available.
- Offline scan: no network access, no public advisory checks for pinned dependency versions, and no execution of application code or interaction with a live cluster.
- Server-side Kubernetes API server admission behavior (ownerReference finalizer validation, Role/RoleBinding escalation prevention) could not be source-verified because k8s.io/apiserver is not part of the module graph; dependency semantics were instead verified from the local Go module cache for the pinned versions.
- Excluded unspecified: Files named ._ prefix are macOS AppleDouble binary resource-fork metadata, not source; excluded from review scope.
- Excluded unspecified: go.sum was parsed for pinned module versions but not line-reviewed; it contains no executable logic.

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 7 |
| Severity mix | medium: 1, low: 6 |
| Confidence mix | high: 4, medium: 3 |
| Coverage | complete |
| Validation mode | Parent validation against local source plus dependency-semantics verification in the Go module cache |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

namespace-watch is a cluster-wide controller that lists and watches all Namespaces and, for each Active namespace whose name does not match a configured skip-prefix, creates six hardcoded default resources (LimitRange, ResourceQuota, ServiceAccount, Role, RoleBinding, NetworkPolicy) through a dynamic client, each owned by the Namespace object. The security-relevant boundary is namespace API metadata crossing into the controller's enqueue filter, template rendering, ownerReferences, and the dynamic Create path, while the controller holds cluster-wide create-only RBAC. The primary actors are cluster principals able to create namespaces, an operator deploying the controller, and an attacker who compromises the controller pod or its image.

### Assets

- Cluster RBAC integrity and the controller ServiceAccount identity
- Correct placement and content of resources created in every namespace
- Availability of system and shared namespaces (e.g. kube-system, default) with respect to injected quotas and limit ranges
- Per-namespace tenant guardrails (ResourceQuota, LimitRange) the product is meant to install

### Trust Boundaries

- Namespace API metadata (name, phase, UID) produced by cluster users into the controller's informer, enqueue filter, and dynamic-client request path
- Operator-controlled CLI flags and kubeconfig resolution into the controller process identity
- Cluster RBAC grants (ClusterRole/ClusterRoleBinding) into API-server authorization for all controller writes
- Container image build pipeline (build context, image reference) into runtime execution under a cluster-scoped ServiceAccount

### Attacker Capabilities

- Create (and delete/recreate) Namespaces as an ordinary cluster tenant
- Edit the Deployment manifest, flags, and image reference with write access to the controller namespace (operator path)
- Repoint a mutable image tag or compromise the build pipeline
- Achieve code execution inside the controller pod and use the automounted ServiceAccount token

### Security Objectives

- Created resources land only in intended namespaces with exactly the intended content
- System and shared namespaces are excluded from tenant-default injection, and the exclusion control fails closed
- The controller's granted RBAC is no broader than the six fixed resources it creates
- The controller acts only under its own scoped identity, and the executed artifact is integrity-pinned

### Assumptions

- The shipped deploy/deploy.yaml is representative of a real deployment of this image
- The API server validates namespace names as RFC 1123 labels, so crafted-name path/URL injection through the dynamic client is out of reach
- Client behavior claims were verified against the pinned dependency sources in the local Go module cache (k8s.io/client-go v0.30.0, k8s.io/apimachinery v0.30.0)

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Namespace exclusion is prefix-only and replayed over all pre-existing namespaces, so restrictive quotas and limits land in shared and system-like namespaces](#finding-1) | medium | high | inline below |
| [Dockerfile copies the entire build context with no .dockerignore, so build-time secrets are baked into intermediate layers and build cache](#finding-2) | low | high | inline below |
| [Deployed ClusterRole lacks the bind and namespaces/finalizers permissions that the controller's own Role, RoleBinding, and blockOwnerDeletion ownerReferences require; permanent validation errors requeue unbounded](#finding-3) | low | medium | inline below |
| [Cluster-privileged Deployment pulls a mutable :latest image with no registry host or digest pinning](#finding-4) | low | high | inline below |
| [buildRESTConfig silently falls back to ambient kubeconfig credentials when in-cluster config is unavailable](#finding-5) | low | medium | inline below |
| [Delete/recreate UID race with controller ownerReferences and Add-only reconciliation can permanently drop namespace guardrails](#finding-6) | low | medium | inline below |
| [-skip-prefixes is split without trimming or validation: an empty element silently disables all provisioning, and a whitespace-padded element silently removes that namespace protection](#finding-7) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Namespace exclusion is prefix-only and replayed over all pre-existing namespaces, so restrictive quotas and limits land in shared and system-like namespaces

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Enqueue filter, Add-only handler registration order, tolerate-AlreadyExists sync, and the shipped default flag value are all directly readable in repository source; the initial-LIST Add replay was additionally verified in the pinned k8s.io/client-go v0.30.0 sources (shared_informer.go dispatches initial-list items through OnAdd). |
| Category | broken-access-control |
| CWE | CWE-863 |
| Affected lines | internal/controller/controller.go:82-93, internal/controller/controller.go:53-61, internal/controller/controller.go:123-146, internal/defaults/defaults.go:77-98, cmd/namespace-watch/main.go:30-31, deploy/deploy.yaml:61-63 |

#### Summary

The only control that keeps tenant defaults out of protected namespaces is an exact name-prefix check at enqueue time (default -skip-prefixes="kube-"). The informer registers its AddFunc before Run, so client-go replays the initial LIST of all Active namespaces as Add events on every controller start, and the namespace "default" plus any add-on or system-like namespace whose name does not begin with "kube-" (for example istio-system, ingress-nginx, kubernetes-dashboard) is stamped with a fixed production-quota ResourceQuota and a max-capped LimitRange. The injection is sticky because sync tolerates AlreadyExists and never updates or removes existing objects.

#### Root Cause

The protected-namespace decision is made once at enqueue time using only exact name prefixes, and pre-existing namespaces re-enter that path as Add events during the informer's initial LIST replay, so the filter's coverage is defined by naming convention rather than by namespace identity or intent.

**Code evidence 1** — `internal/controller/controller.go:86-90`

```
	for _, prefix := range c.skipPrefixes {
		if strings.HasPrefix(ns.Name, prefix) {
			return
		}
	}
```

**Code evidence 2** — `internal/controller/controller.go:53-61`

```
	_, err := c.informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			ns, ok := obj.(*corev1.Namespace)
			if !ok {
				return
			}
			c.enqueue(ns)
		},
	})
```

#### Validation

Confirmed by reading the full reconcile path and the shipped deployment, and by verifying in the pinned client-go v0.30.0 module source that the shared informer delivers initial-list items to registered handlers through OnAdd, so every Active namespace is enqueued on controller start.

Validation method: Source review plus dependency-source verification in the Go module cache (k8s.io/client-go v0.30.0).

**Code evidence 1** — `internal/controller/controller.go:86-90`

```
	for _, prefix := range c.skipPrefixes {
		if strings.HasPrefix(ns.Name, prefix) {
			return
		}
	}
```

**Code evidence 2** — `internal/controller/controller.go:134-136`

```
		_, err := c.dynamic.Resource(res.GVR).Namespace(ns.Name).
			Create(ctx, res.Body, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
```

Evidence:
- Shipped args are only -v=2, so the default -skip-prefixes="kube-" is the effective filter (deploy/deploy.yaml:62-63).
- Enqueue is the sole consumer of skipPrefixes; sync performs no second check.

Counterevidence and remaining uncertainty:
- ResourceQuota creation is rejected when current usage already exceeds the hard values, so some hot namespaces reject the injection.
- The injected NetworkPolicy is inert unless pods carry the app=myapp label, limiting the blast radius to quota and limit-range effects.
- Operators can widen -skip-prefixes, which mitigates but does not fix the naming-convention control.

#### Dataflow

Namespace objects from the cluster-wide watch (including the initial LIST replay on every start) flow through AddFunc into the prefix-only filter, then into sync, which issues six Create calls that stamp quotas, limit ranges, and RBAC objects into whatever Active namespace survived the filter.

#### Reachability

Automatic at every controller start or redeploy for pre-existing Active namespaces; additionally reachable by any principal able to create a namespace whose name does not match a configured prefix, which the controller then provisions with the same fixed defaults.

#### Severity

**Medium** — Namespace-wide quotas (pods 100, requests.cpu 20) and per-container caps (max cpu 2, memory 2Gi) in shared or platform namespaces cause sustained rejection of pod, service, and PVC creation for workloads outside the product's intended scope. Likelihood is elevated because the default namespace is always affected on first and every start in the shipped configuration. High impact with elevated likelihood calibrates to medium rather than high because some exposure is design intent (any namespace not skipped is provisioned) and hot namespaces with usage above the hard values reject the quota at creation.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Replace name-prefix exclusion with an explicit opt-in/opt-out namespace label or annotation, re-check the filter inside sync against the live object, skip namespaces that pre-date controller startup unless opted in, and enumerate protected namespaces (default included) in deployment args as defense in depth.

<a id="finding-2"></a>

### [2] Dockerfile copies the entire build context with no .dockerignore, so build-time secrets are baked into intermediate layers and build cache

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The absence of .dockerignore anywhere in the scan root and the COPY . . instruction are directly verifiable. |
| Category | information-exposure |
| CWE | CWE-497 |
| Affected lines | Dockerfile:1-6, Dockerfile:8-10 |

#### Summary

The build stage runs COPY . . with no .dockerignore present. Any kubeconfig, token file, .env, or VCS material sitting in the build directory at build time is copied into the build-stage layer and remains retrievable from builder cache, pushed cache stages, or intermediate-stage artifacts, even though the final distroless stage copies only the binary. The current context contains no secrets, so this is exposure shaped by construction rather than an active leak.

#### Root Cause

The whole working tree becomes build-layer content with no ignore file to exclude secrets, so layer history inherits whatever happened to be in the directory at build time.

**Code evidence 1** — `Dockerfile:2-6`

```
FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
```

#### Validation

Confirmed by directory listing (no ignore file of any kind) and the Dockerfile itself; the two-stage build limits the shipped image but not cache or intermediate artifacts.

Validation method: Build-pipeline review.

**Code evidence 1** — `Dockerfile:2-6`

```
FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
```

Evidence:
- Final stage copies only /namespace-watch, so runtime images do not carry the context contents.

Counterevidence and remaining uncertainty:
- The present context contains no secret material, so this is shaped-by-construction exposure rather than an active leak.

#### Dataflow

A secret file in the build directory is copied into the build-stage layer and remains recoverable from pushed cache or intermediate stages, reaching anyone with registry-cache read access.

#### Reachability

Requires a build to run while a secret is present in the tree plus access to build cache or published intermediate layers.

#### Severity

**Low** — No secret is currently in the context and the shipped final image contains only the binary; impact requires a future build performed while a secret sits in the tree plus reader access to cache or intermediate layers. Low likelihood and conditional impact calibrate to low.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Add a .dockerignore (at minimum .git, bin, .env, kubeconfig\*, dotfiles) or replace COPY . . with an explicit allowlist of go.mod, go.sum, cmd/, and internal/; build from a pristine VCS checkout in CI and do not push intermediate stages.

<a id="finding-3"></a>

### [3] Deployed ClusterRole lacks the bind and namespaces/finalizers permissions that the controller's own Role, RoleBinding, and blockOwnerDeletion ownerReferences require; permanent validation errors requeue unbounded

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The grant set, the emitted object content, and the unbounded default requeue branch are all confirmed in repository and pinned-dependency source; the triggering admission checks live in k8s.io/apiserver, which is not in the module graph, so their activation depends on cluster version and enabled plugins. |
| Category | security-configuration |
| CWE | CWE-269 |
| Affected lines | deploy/deploy.yaml:16-28, internal/defaults/defaults.go:112-153, internal/defaults/defaults.go:226-231, internal/controller/controller.go:100-119 |

#### Summary

The controller creates a Role granting configmaps get/list, a RoleBinding binding it, and objects carrying blockOwnerDeletion=true controller ownerReferences to Namespaces, while its ClusterRole grants create-only verbs and namespaces get/list/watch. On API servers enforcing the standard escalation-prevention, role-bind, and ownerReference finalizer admission checks, the ServiceAccount cannot legally create these objects, so every sync fails with a non-transient error that processNext classifies as retryable and requeues forever with no max-requeue bound.

#### Root Cause

The manifest was written against verbs only, but the objects the code emits also require escalation-prevention, role-bind, and ownerReference finalizer admission to be satisfied, and the reconcile loop treats permanent validation rejections as transient.

**Code evidence 1** — `deploy/deploy.yaml:20-25`

```
  - apiGroups: [""]
    resources: ["limitranges", "resourcequotas", "serviceaccounts"]
    verbs: ["create"]
  - apiGroups: ["rbac.authorization.k8s.io"]
    resources: ["roles", "rolebindings"]
    verbs: ["create"]
```

**Code evidence 2** — `internal/defaults/defaults.go:119-125`

```
		"rules": []interface{}{
			map[string]interface{}{
				"apiGroups": []interface{}{""},
				"resources": []interface{}{"configmaps"},
				"verbs":     []interface{}{"get", "list"},
			},
		},
```

**Code evidence 3** — `internal/controller/controller.go:114-117`

```
		default:
			klog.ErrorS(err, "sync failed, requeueing", "namespace", key)
			c.queue.AddRateLimited(key)
```

#### Validation

Repository evidence confirms the grant set, the emitted Role/RoleBinding content, the blockOwnerDeletion=true controller references, and the unbounded requeue branch. The API-server admission requirements behind the break are established from Kubernetes documented behavior, not repository source, and are recorded as a proof gap.

Validation method: Source review of manifest and templates plus documented admission-control behavior; no cluster was available for runtime confirmation.

**Code evidence 1** — `internal/controller/controller.go:114-117`

```
		default:
			klog.ErrorS(err, "sync failed, requeueing", "namespace", key)
			c.queue.AddRateLimited(key)
```

Evidence:
- ClusterRole and ClusterRoleBinding are the only grants to the controller ServiceAccount (deploy/deploy.yaml:12-41).
- processNext only forgets NotFound errors; everything else is AddRateLimited without a bound.

Counterevidence and remaining uncertainty:
- Clusters whose version or admission configuration does not enforce these checks are unaffected, which is why confidence is medium rather than high.

#### Dataflow

No external attacker is required: the shipped RBAC plus emitted object content deterministically produces permanent API errors that the workqueue retries forever. A namespace owner pre-creating one of the six fixed names with an owner the SA cannot finalize reproduces the same unbounded requeue for that key.

#### Reachability

Automatic in any enforcing deployment; attacker-assisted variant is reachable by an ordinary namespace tenant but adds nothing beyond the self-inflicted path.

#### Severity

**Low** — Impact is product failure with log and worker churn rather than privilege compromise: the same RBAC mismatch keeps the controller below its own intended grants, and API-server enforcement also keeps it below escalation. The server-side enforcement point could not be source-verified, and clusters without the relevant admission behavior are unaffected, so severity is capped at low with medium confidence.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Either align the grant set with what the API server requires for the intended objects (bind on the specific resourceNames, and namespaces update or the finalizers subresource if blockOwnerDeletion is required) or emit objects that do not need those permissions (drop blockOwnerDeletion; ship the Role/RoleBinding via the deployment instead of the controller); classify non-transient validation errors as terminal with a metric instead of requeueing without a bound.

<a id="finding-4"></a>

### [4] Cluster-privileged Deployment pulls a mutable :latest image with no registry host or digest pinning

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The image reference, the ServiceAccount binding, and the grant set are directly readable in the manifest; no integrity or admission control exists anywhere in the repository. |
| Category | supply-chain |
| CWE | CWE-494 |
| Affected lines | deploy/deploy.yaml:57-63, deploy/deploy.yaml:12-41 |

#### Summary

The Deployment runs with a ServiceAccount bound to a cluster-wide ClusterRole but references image namespace-watch:latest, an implicit-registry mutable tag with no digest and no imagePullPolicy. Whoever can repoint that tag in the registry or node image store substitutes the binary that receives cluster-wide create rights in every namespace, collapsing the separation between artifact integrity and cluster-wide RBAC.

#### Root Cause

Artifact identity is expressed as a mutable tag for a workload whose identity carries cluster-wide write capability, so tag substitution silently inherits the RBAC.

**Code evidence 1** — `deploy/deploy.yaml:58-61`

```
      serviceAccountName: namespace-watch
      containers:
        - name: namespace-watch
          image: namespace-watch:latest
```

**Code evidence 2** — `Dockerfile:8-10`

```
FROM gcr.io/distroless/static:nonroot
COPY --from=build /namespace-watch /namespace-watch
ENTRYPOINT ["/namespace-watch"]
```

#### Validation

Confirmed directly in deploy/deploy.yaml together with the ClusterRole grant set it inherits. The pod-level hardening that is present (nonroot, read-only rootfs, no privilege escalation) reduces post-execution risk but does not verify artifact identity.

Validation method: Manifest review.

**Code evidence 1** — `deploy/deploy.yaml:58-61`

```
      serviceAccountName: namespace-watch
      containers:
        - name: namespace-watch
          image: namespace-watch:latest
```

Evidence:
- No digest, registry host, imagePullPolicy, or admission-based verification appears anywhere in the repository.

Counterevidence and remaining uncertainty:
- Dev clusters typically side-load such images instead of pulling, which shrinks the tag-mutation surface; an external admission controller outside this repository could already enforce pinning.

#### Dataflow

A tag mutation in whatever store the kubelet resolves flows unchanged into container execution mounted with the cluster-privileged ServiceAccount token.

#### Reachability

Requires control of the registry, CI pipeline, or node image store; not reachable from ordinary cluster user privileges.

#### Severity

**Low** — Impact if exploited is near cluster-takeover, but likelihood is low: exploitation requires registry, CI, or node image-store control rather than cluster access, and the unqualified placeholder name limits public-registry squatting. High impact with low likelihood calibrates to low.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Reference the image by registry domain plus immutable digest (or an immutable version tag with imagePullPolicy IfNotPresent), pin the Dockerfile base image by digest, enforce digest pinning for privileged namespaces with an admission policy, and sign artifacts with verification at admission.

<a id="finding-5"></a>

### [5] buildRESTConfig silently falls back to ambient kubeconfig credentials when in-cluster config is unavailable

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The fallback chain and the swallowed in-cluster error are directly in source; whether any real deployment exposes ambient credentials depends on environments outside the repository. |
| Category | credential-management |
| CWE | CWE-1188 |
| Affected lines | cmd/namespace-watch/main.go:65-76 |

#### Summary

When the -kubeconfig flag is empty and rest.InClusterConfig fails, the error is swallowed and the controller falls through to the default client loading rules, silently adopting $KUBECONFIG or ~/.kube/config. Any environment where the image runs without in-cluster config (automount disabled, host-network pod, kind/minikube, CI, a developer machine) then executes the controller's privileged writes under an ambient, often much broader, identity with no log of which credential was chosen and no fail-closed.

#### Root Cause

Identity selection fails open: any failure of the in-cluster configuration silently switches the controller's identity to whatever ambient kubeconfig exists, with no signal and no explicit opt-in.

**Code evidence 1** — `cmd/namespace-watch/main.go:69-75`

```
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
```

#### Validation

Confirmed by reading the full resolution order. The in intended-cluster path (serviceAccountName set, automount default true) the fallback is never reached, which bounds exposure to non-standard executions of the same image.

Validation method: Source review of the credential resolution chain.

**Code evidence 1** — `cmd/namespace-watch/main.go:69-75`

```
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
```

Evidence:
- The fallback is the standard client-go loading-rules chain, which also honors exec-credential plugins from the loaded file.

Counterevidence and remaining uncertainty:
- No repository file references ~/.kube or KUBECONFIG; exploitation requires an operator or environment to mount a credential the image can read.

#### Dataflow

An ambient credential file read by the default loading rules becomes the effective identity for all six dynamic Create calls, silently crossing from the controller's scoped identity to a broader user identity.

#### Reachability

Reached whenever the image runs outside its intended Deployment with a readable kubeconfig present; deployer- and environment-controlled.

#### Severity

**Low** — This is a credential-confusion trust-boundary weakness rather than an exploitable remote path: in the shipped Deployment the fallback is dead code, reaching it requires an operator or environment to place a broader credential where the image can read it, and the -kubeconfig flag's trust is standard operator behavior. Low with concrete boundary-crossing shape calibrates to low.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Gate the ambient fallback behind an explicit out-of-cluster opt-in flag or environment variable, refuse it when a service-account root or pod environment is detected, and log the resolved cluster and identity at startup so an operator can always see which credential is in use.

<a id="finding-6"></a>

### [6] Delete/recreate UID race with controller ownerReferences and Add-only reconciliation can permanently drop namespace guardrails

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The NewControllerRef semantics were verified directly against the pinned apimachinery v0.30.0 source, and the tolerate-AlreadyExists plus Add-only code paths are in repository source; the end-to-end loss depends on garbage-collector timing that cannot be reproduced from source alone. |
| Category | concurrency-state-handling |
| CWE | CWE-367 |
| Affected lines | internal/defaults/defaults.go:226-231, internal/controller/controller.go:123-146, internal/controller/controller.go:53-61, cmd/namespace-watch/main.go:27-28 |

#### Summary

OwnedBy attaches a metav1.NewControllerRef ownerReference (controller=true and blockOwnerDeletion=true, verified in apimachinery v0.30.0) pointing at the Namespace UID held in the informer cache. If a namespace is deleted and recreated with the same name before the garbage collector drains the old dependents, the recreated namespace's sync hits AlreadyExists on the stale-UID objects and tolerates it; the GC then deletes those dependents, and because only Add events reconcile and resync is disabled by default, nothing re-creates them, silently removing the quota and limit-range guardrails until the next controller restart.

#### Root Cause

Ownership is captured from a cached Namespace UID at create time and never repaired afterwards, while reconciliation is driven only by Add events with resync disabled by default.

**Code evidence 1** — `internal/defaults/defaults.go:227-230`

```
	u.SetNamespace(owner.Name)
	u.SetOwnerReferences([]metav1.OwnerReference{
		*metav1.NewControllerRef(owner, corev1.SchemeGroupVersion.WithKind("Namespace")),
	})
```

**Code evidence 2** — `internal/controller/controller.go:136-143`

```
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("creating %s/%s in namespace %s: %w",
				res.Body.GetKind(), res.Body.GetName(), ns.Name, err)
		}
		if err == nil {
			klog.InfoS("created default resource",
```

#### Validation

Verified NewControllerRef field semantics in the pinned dependency source and the absence of any owner-reference repair path in repository source; the race itself is a reasoned interleaving of documented GC behavior with the confirmed code paths.

Validation method: Dependency-source verification plus control-flow review of sync and event handlers.

**Code evidence 1** — `internal/defaults/defaults.go:227-230`

```
	u.SetNamespace(owner.Name)
	u.SetOwnerReferences([]metav1.OwnerReference{
		*metav1.NewControllerRef(owner, corev1.SchemeGroupVersion.WithKind("Namespace")),
	})
```

Evidence:
- No UpdateFunc or DeleteFunc is registered anywhere in the controller.
- The informer indexer read in sync cannot distinguish a recreated namespace with the same name.

Counterevidence and remaining uncertainty:
- In smaller clusters the GC usually drains dependents before recreation, after which creates succeed with fresh ownerReferences.
- A controller restart replays the initial list and restores missing defaults, so the loss is not permanent across restarts.

#### Dataflow

A principal with delete and create rights on its own namespace deletes and recreates it around the GC window; the controller's create-time UID capture plus AlreadyExists tolerance converts the race into permanently unreconciled default resources.

#### Reachability

Reachable by a namespace owner through ordinary delete/recreate churn; no elevated privilege required, but timing-dependent.

#### Severity

**Low** — Impact is medium (tenant guardrails silently absent in one namespace, undetected) but likelihood is low: it requires a delete/recreate race against GC timing, and a controller restart heals the state via the initial-list replay. Medium impact with low likelihood calibrates to low.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

On AlreadyExists, compare the existing object's controller ownerReference UID against the live namespace and repair or requeue when it is stale; register Update handlers or enable a bounded resync so drift self-heals; consider ownerReferences without blockOwnerDeletion since namespace deletion already cascades to contained resources.

<a id="finding-7"></a>

### [7] -skip-prefixes is split without trimming or validation: an empty element silently disables all provisioning, and a whitespace-padded element silently removes that namespace protection

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Go standard-library semantics of strings.Split and strings.HasPrefix with an empty prefix are deterministic, and the enqueue filter is the sole consumer of the parsed value; both are directly in repository source. |
| Category | input-validation |
| CWE | CWE-20 |
| Affected lines | cmd/namespace-watch/main.go:53-54, internal/controller/controller.go:82-93, cmd/namespace-watch/main.go:30-31 |

#### Summary

strings.Split on the raw flag value produces empty elements for empty values, trailing commas, or doubled commas, and never trims whitespace. An empty prefix matches every namespace name in the enqueue filter, so the controller silently stops provisioning everything (fail-closed no-op with no log or error). A padded element such as the second entry in "kube-, istio-" can never match a valid RFC 1123 name, so that namespace family is silently unprotected and receives the production quota and limit range (fail-open).

#### Root Cause

The exclusion list is consumed verbatim with no normalization, so representation variants of the flag flip the control in both directions: an empty element matches everything and skips everything, while an unrecoverable padded element matches nothing and protects nothing.

**Code evidence 1** — `cmd/namespace-watch/main.go:53-54`

```
	ctrl := controller.New(dyn, informerFactory,
		strings.Split(*skipPrefixes, ","))
```

**Code evidence 2** — `internal/controller/controller.go:86-90`

```
	for _, prefix := range c.skipPrefixes {
		if strings.HasPrefix(ns.Name, prefix) {
			return
		}
	}
```

#### Validation

Confirmed by direct reading of the flag parsing and the sole consumer of skipPrefixes; Go semantics for Split and HasPrefix with empty and padded elements are deterministic. Note: an earlier worker draft inverted the empty-element direction (claiming quotas would be injected into system namespaces); parent re-derivation of the enqueue semantics corrects this: an empty prefix causes HasPrefix to return true, which skips the namespace, so the empty-element face is a silent no-op, and the fail-open face comes from padded elements.

Validation method: Source review with careful re-derivation of the filter direction; corrected a worker-reported inversion.

**Code evidence 1** — `internal/controller/controller.go:86-90`

```
	for _, prefix := range c.skipPrefixes {
		if strings.HasPrefix(ns.Name, prefix) {
			return
		}
	}
```

Evidence:
- The shipped default "kube-" is itself safe; both faces require an operator to edit args.

Counterevidence and remaining uncertainty:
- The no-op face disables provisioning rather than injecting into system namespaces, so the two faces have different impacts and neither reaches beyond operator control.

#### Dataflow

Deployment args flow through strings.Split into the enqueue filter unmodified; representation variants of the exclusion list silently broaden or disable the control.

#### Reachability

Requires write access to the controller Deployment spec, i.e. the operator path; no runtime cluster user can reach the flag.

#### Severity

**Low** — Only an operator who can edit the Deployment args can reach this, and that principal already controls the controller; the realistic damage is self-inflicted quota injection into protected namespaces or a silent product no-op. Operator-only reachability caps severity at low despite the concrete availability potential.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Trim whitespace per element, drop empty elements, fail startup when the resulting prefix set is empty or contains entries that cannot match a namespace name, log the effective filter at startup, and re-assert the filter inside sync.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| CLI flags, kubeconfig resolution, and client bootstrap (cmd/namespace-watch/main.go) | not recorded | Reported | No additional canonical notes were recorded. |
| Informer, work queue, and namespace reconcile path (internal/controller/controller.go) | not recorded | Reported | No additional canonical notes were recorded. |
| Default resource templates and ownerReferences (internal/defaults/defaults.go) | not recorded | Reported | No additional canonical notes were recorded. |
| Deployment RBAC, pod hardening, and image reference (deploy/deploy.yaml) | not recorded | Reported | No additional canonical notes were recorded. |
| Container build pipeline and dependency manifests (Dockerfile, go.mod, go.sum) | not recorded | Reported | No additional canonical notes were recorded. |
| Controller and defaults test suites (internal/controller/controller_test.go, internal/defaults/defaults_test.go) | not recorded | No issue found | No additional canonical notes were recorded. |

## Open Questions And Follow Up

- Server-side enforcement of the ownerReference finalizer permission (blockOwnerDeletion=true) and of Role/RoleBinding escalation prevention depends on the API server version and enabled admission plugins; k8s.io/apiserver is not in the module graph, so the rejection behavior behind the RBAC-mismatch finding was not source-verifiable from this repository.
- Dependency currency (k8s.io/\* v0.30.0, golang.org/x/net v0.23.0) was not checked against public advisories because this scan is offline; an online govulncheck pass is recommended.
