# Security Review: namespace-watch-qwen3.8-27b-uncensored

## Scope

Repository-wide standard security scan of the namespace-watch-qwen3.8-27b-uncensored directory (non-Git scan root). All Go source (main.go, internal/config, internal/provisioner, internal/watcher, tests), deployment manifests (deploy/namespace-watcher.yaml), Dockerfile, Makefile, go.mod/go.sum, and README.md were reviewed. The compiled binary bin/namespace-watcher was excluded from static analysis. No SECURITY.md policy exists at the scan root or in any subdirectory.

- Scan mode: repository
- Target kind: directory_snapshot
- Target ID: local:sha256:2b53081f1430b2ba17ed27f5eb505bf9e31e3c482ff56e7a2e7aafbc06f901db
- Snapshot digest: codex-security-snapshot/v1:sha256:95c767aa86ea26eda425733a9536af80166b467e1a43aa7e3a2774c09d09a845
- Inventory strategy: directory
- Included paths: .
- Excluded paths: none
- Runtime or test status: not recorded

Limitations and exclusions:
- Excluded bin/namespace-watcher: Compiled Go binary; excluded from static source analysis (no decompilation performed). Source-level audit covers the equivalent Go sources.

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 7 |
| Severity mix | high: 1, medium: 2, low: 4 |
| Confidence mix | high: 5, medium: 2 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

namespace-watcher is a cluster-wide Kubernetes operator: it watches namespace creation and provisions security-default resources (ResourceQuota, LimitRange, read-only RBAC triple, NetworkPolicy) into each new namespace. The cluster's per-tenant resource containment and default network isolation rely on it. The primary external attacker is a cluster tenant who can create namespaces (and typically manage resources within them). The watcher itself runs with a powerful ClusterRole (create roles/rolebindings in any namespace), so its state machine, event pipeline, and deployment are security-relevant. Trust boundaries: (1) Kubernetes API namespace objects to the unstructured resources the watcher builds; (2) the watcher's ServiceAccount credentials to the API server; (3) the image/manifest supply chain. Security objectives: every existing namespace carries the default quota/isolation controls; defaults are applied promptly and idempotently; the watcher runs least-privilege and stays available.

### Assets

- per-namespace ResourceQuota/LimitRange (cross-tenant resource containment)
- per-namespace NetworkPolicy (default network isolation)
- per-namespace read-only RBAC defaults
- watcher availability (the control applicator)
- cluster RBAC surface reachable via the watcher ServiceAccount

### Trust Boundaries

- Kubernetes API (namespace objects, names) -\> watcher unstructured resource templates and dynamic-client create path
- watcher ServiceAccount (ClusterRole: watch namespaces cluster-wide; create/get/list six resource types in any namespace) -\> API server
- container image and deployment manifest -\> watcher process

### Attacker Capabilities

- tenant: create namespaces; commonly also create/delete resources inside own namespace
- tenant: sustain namespace-creation bursts to load the watcher queue
- cluster-admin-level: replace the :latest image or mutate the deployment (supply chain)
- in-cluster: observe namespace lifecycle timing

### Security Objectives

- every namespace that exists is provisioned with the default security resources
- provisioning is prompt, idempotent, and resilient to transient API failures
- the watcher ServiceAccount holds only the privileges the code exercises
- the watcher remains available under adversarial namespace-creation load

### Assumptions

- the cluster relies on this watcher (not a separate admission controller) to apply tenant defaults, per the README's stated purpose
- namespace names are validated by the API server to RFC 1123 form, so no injection surface exists in the name field
- compensating cluster-level quotas/admission controls, if any, are outside this repository

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Security defaults are never reapplied to a namespace that is deleted and recreated with the same name](#finding-1) | high | high | inline below |
| [A failed resource creation permanently marks the namespace provisioned with no retry](#finding-2) | medium | high | inline below |
| [Pre-created same-named resources masquerade as the watcher defaults (no ownership verification)](#finding-3) | medium | medium | inline below |
| [ClusterRole grants cluster-wide get/list on six resource types the code never uses](#finding-4) | low | high | inline below |
| [Default NetworkPolicy only constrains pods labeled app=myapp; typical workloads in new namespaces remain unrestricted](#finding-5) | low | high | inline below |
| [Sustained namespace-creation bursts block the informer event loop and delay all default provisioning](#finding-6) | low | medium | inline below |
| [Deployment uses a mutable :latest image tag for a highly privileged operator](#finding-7) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Security defaults are never reapplied to a namespace that is deleted and recreated with the same name

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | high |
| Confidence rationale | All code paths were read in full: the `done` map is written before the object loop (provisioner.go:60), is only ever read (no `delete(p.done, ...)` anywhere), `onAdd` checks `IsDone` and returns (watcher.go:177-180), and only an `AddFunc` handler is registered (watcher.go:100-104). Grep across the repository confirms no delete/update informer handler and no map reset. The only runtime assumption is the tenant RBAC profile (create+delete on namespaces), which is recorded as a precondition. |
| Category | insecure_design |
| CWE | CWE-1188 |
| Affected lines | internal/provisioner/provisioner.go:53-80, internal/watcher/watcher.go:156-194, internal/watcher/watcher.go:100-104, deploy/namespace-watcher.yaml:87-90 |

#### Summary

The watcher keys its provisioning state in an in-memory `done` map by namespace name and never clears it; no namespace delete handler is registered and no reconciliation exists. `onAdd` short-circuits when `provisioner.IsDone(name)` is true, so a namespace that was provisioned once, then deleted and recreated with the same name, is skipped for the watcher's entire lifetime. Its ResourceQuota, LimitRange, and NetworkPolicy were garbage-collected with the deleted namespace, leaving the recreation without any of the cluster's per-tenant containment defaults. With the deployed default (no `--backfill` in deploy/namespace-watcher.yaml), the escape also survives a process restart because the recreated namespace is captured into the `known` set at startup and treated as pre-existing.

#### Root Cause

The security invariant is that every namespace that exists carries the provisioner's default ResourceQuota, LimitRange, and NetworkPolicy. The state that enforces one-time provisioning is an in-memory map keyed by namespace name (`done`), which is written before the creates and is never reconciled against the actual namespace lifecycle: there is no delete handler, no periodic re-list, and no retry. Deleting and recreating a namespace destroys the six objects (Kubernetes garbage collection) while the `done` entry survives, so the next Add event is discarded and the invariant is silently broken for the remainder of the process lifetime.

**Provision marks the namespace done before creating any resource** — `internal/provisioner/provisioner.go:53-61`

The namespace name is inserted into `done` before the six resource creates run. Nothing ever removes the entry: no namespace delete handler exists and no code path calls `delete` on the map, so the state outlives the namespace itself.

```go
func (p *Provisioner) Provision(ctx context.Context, namespace string) error {
	p.mu.Lock()
	if _, ok := p.done[namespace]; ok {
		p.mu.Unlock()
		p.log.Debug("skipping already-provisioned namespace", "namespace", namespace)
		return nil
	}
	p.done[namespace] = struct{}{}
	p.mu.Unlock()
```

**onAdd skips any namespace the provisioner considers done** — `internal/watcher/watcher.go:176-186`

When the recreated namespace's Add event fires, `IsDone(name)` returns true from the first lifecycle, so the watcher returns without provisioning. The recreation is never enqueued.

```go
	w.queueMu.Lock()
	if w.provisioner.IsDone(name) {
		w.queueMu.Unlock()
		return
	}
	if _, inflight := w.inflight[name]; inflight {
		w.queueMu.Unlock()
		return
	}
	w.inflight[name] = struct{}{}
	w.queueMu.Unlock()
```

**Only an Add event handler is registered** — `internal/watcher/watcher.go:100-104`

No `DeleteFunc` (or `UpdateFunc`) is registered, so namespace deletion produces no state reset and no re-provisioning trigger. A delete handler that clears the `done` entry is the missing control.

```go
	if _, err := w.nsInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: w.onAdd,
	}); err != nil {
		return fmt.Errorf("add namespace event handler: %w", err)
	}
```

**Deployment does not enable backfill** — `deploy/namespace-watcher.yaml:87-90`

The shipped manifest runs in new-only mode. After a watcher restart, the recreated namespace is captured into the `known` set (watcher.go:139-152) and skipped, so the escape persists across restarts unless an operator switches to `--backfill`.

```yaml
          args:
            - --log-level=info
            # Add --backfill to also provision namespaces that already exist
            # when the watcher first starts.
```

#### Validation

Traced the namespace lifecycle through the source: (1) first lifecycle creates the six objects and leaves `done[name]` set; (2) namespace deletion removes the objects via cluster garbage collection and fires no handler, because only `AddFunc` is registered (watcher.go:100-104) and grep confirms no `DeleteFunc`/`UpdateFunc` and no `delete(p.done, ...)` anywhere; (3) recreation fires Add for the new object, `onAdd` calls `provisioner.IsDone(name)` (watcher.go:177) which returns true, and the function returns before enqueueing; (4) on process restart without `--backfill`, `captureExisting` records the namespace in `known` (watcher.go:143-149) and `onAdd` skips known namespaces (watcher.go:169-171). No code path re-provisions the namespace.

Validation method: static source trace of the full watcher/provisioner state machine

**Provision marks the namespace done before creating any resource** — `internal/provisioner/provisioner.go:53-61`

The namespace name is inserted into `done` before the six resource creates run. Nothing ever removes the entry: no namespace delete handler exists and no code path calls `delete` on the map, so the state outlives the namespace itself.

```go
func (p *Provisioner) Provision(ctx context.Context, namespace string) error {
	p.mu.Lock()
	if _, ok := p.done[namespace]; ok {
		p.mu.Unlock()
		p.log.Debug("skipping already-provisioned namespace", "namespace", namespace)
		return nil
	}
	p.done[namespace] = struct{}{}
	p.mu.Unlock()
```

**onAdd skips any namespace the provisioner considers done** — `internal/watcher/watcher.go:176-186`

When the recreated namespace's Add event fires, `IsDone(name)` returns true from the first lifecycle, so the watcher returns without provisioning. The recreation is never enqueued.

```go
	w.queueMu.Lock()
	if w.provisioner.IsDone(name) {
		w.queueMu.Unlock()
		return
	}
	if _, inflight := w.inflight[name]; inflight {
		w.queueMu.Unlock()
		return
	}
	w.inflight[name] = struct{}{}
	w.queueMu.Unlock()
```

**Only an Add event handler is registered** — `internal/watcher/watcher.go:100-104`

No `DeleteFunc` (or `UpdateFunc`) is registered, so namespace deletion produces no state reset and no re-provisioning trigger. A delete handler that clears the `done` entry is the missing control.

```go
	if _, err := w.nsInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: w.onAdd,
	}); err != nil {
		return fmt.Errorf("add namespace event handler: %w", err)
	}
```

#### Dataflow

tenant creates namespace -\> AddFunc -\> onAdd -\> IsDone true -\> skip (no provisioning) -\> namespace exists without defaults

- **Source:** tenant Kubernetes API namespace create/delete calls

- **Sink:** provisioner `done` map lookup short-circuiting `onAdd`

- **Outcome:** namespace persists without the cluster's per-tenant containment defaults

#### Reachability

The attacker needs only namespace create and delete permissions and timing control over their own namespace; no watcher compromise or API-server access is required.

- **Attacker:** cluster tenant with namespaces:create and namespaces:delete

- **Entry point:** Kubernetes API namespace lifecycle (create, delete, recreate)

- **Outcome:** permanently un-quota'd, un-isolated namespace usable for cross-tenant resource exhaustion

#### Severity

**High** — Impact is high: the tool's stated purpose is to give every new namespace resource containment (ResourceQuota/LimitRange) and network-isolation defaults; a tenant who defeats it for their namespace can consume unbounded cluster resources (cross-tenant DoS) and operates without the default NetworkPolicy. Likelihood is high: the attack is deterministic and requires only create and delete permissions on namespaces, a common tenant permission set in multi-tenant clusters. The deployed manifest omits `--backfill`, so the escape persists across watcher restarts.

Raise to critical if the cluster relies solely on this watcher for tenant resource containment with no compensating admission controller or cluster-level quotas. Lower to medium if tenants cannot delete namespaces or if compensating cluster-level quotas bound per-namespace consumption.

#### Remediation

Reconcile provisioning state with the namespace lifecycle: register a namespace delete handler that removes the name from the provisioner's done set (or key state by namespace UID and re-provision on Add), and requeue namespaces whose provisioning failed so a transient error cannot permanently skip them.

Tests:
- Test: provision namespace X, delete it, recreate it, and assert the six default resources exist again.
- Test: restart the watcher in new-only mode after a delete/recreate cycle and assert the namespace is provisioned.

Preventive controls:
- Key one-time provisioning state by namespace UID rather than name, or persist it with the namespace (annotation/label) so it is deleted with the namespace.

<a id="finding-2"></a>

### [2] A failed resource creation permanently marks the namespace provisioned with no retry

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Directly proven by the source: `p.done[namespace] = struct{}{}` at provisioner.go:60 precedes the create loop (lines 66-70); errors are joined and returned (lines 71-73) without touching `done`; `watcher.run` (watcher.go:221-230) only logs the error and clears `inflight`; `onAdd` will never re-enqueue the name because `IsDone` is true. |
| Category | insecure_design |
| CWE | CWE-1188, CWE-754 |
| Affected lines | internal/provisioner/provisioner.go:53-79, internal/watcher/watcher.go:221-230 |

#### Summary

`Provision` inserts the namespace into the `done` map before the six create calls and never rolls it back when any create fails. The watcher's `run` only logs the returned error and clears the in-flight marker, so a namespace hit by a transient API error (5xx, timeout, 422 from a malformed default, RBAC denial) is left partially provisioned and is never retried for the process lifetime. For example, if the LimitRange and ResourceQuota creates succeed and the NetworkPolicy create fails, the namespace keeps its quota but silently loses its default network isolation permanently.

#### Root Cause

The violated invariant is that every namespace ends with all six default resources. `Provision` treats the `done` map as a claim token acquired before the work is attempted instead of a completion record, and the caller has no retry mechanism. A mid-loop API failure therefore leaves a permanent partial state that no later event can repair.

**done is marked before the create loop** — `internal/provisioner/provisioner.go:58-73`

The `done` entry is set before any object is created. When a create fails, the joined error is returned but the entry is never removed, so every later `IsDone` check reports the namespace as handled.

```go
	p.done[namespace] = struct{}{}
	p.mu.Unlock()

	p.log.Info("provisioning namespace", "namespace", namespace)

	var errs []error
	for _, obj := range Objects(namespace, p.cfg) {
		if err := p.createObject(ctx, obj); err != nil {
			errs = append(errs, fmt.Errorf("%s/%s: %w", obj.GetKind(), obj.GetName(), err))
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
```

**The worker logs the error and drops it** — `internal/watcher/watcher.go:221-230`

The only consumer of the error is the log line. The `inflight` marker is cleared but the `done` marker is not, and there is no requeue, backoff, or retry, so the failure is terminal for the process lifetime.

```go
func (w *Watcher) run(ctx context.Context, name string) {
	defer func() {
		w.queueMu.Lock()
		delete(w.inflight, name)
		w.queueMu.Unlock()
	}()
	if err := w.provisioner.Provision(ctx, name); err != nil {
		w.log.Error("failed to provision namespace", "namespace", name, "err", err)
	}
```

#### Validation

Verified the full failure path: `createObject` returns any non-AlreadyExists error (provisioner.go:103-124); `Provision` joins and returns it after the loop (provisioner.go:71-73) while `done` remains set (provisioner.go:60); `watcher.run` logs and clears only `inflight` (watcher.go:221-230); `onAdd` then skips the namespace via `IsDone` (watcher.go:177-180). No requeue, timer, or periodic reconciliation exists anywhere in the repository.

Validation method: static source trace of the error path

**done is marked before the create loop** — `internal/provisioner/provisioner.go:58-73`

The `done` entry is set before any object is created. When a create fails, the joined error is returned but the entry is never removed, so every later `IsDone` check reports the namespace as handled.

```go
	p.done[namespace] = struct{}{}
	p.mu.Unlock()

	p.log.Info("provisioning namespace", "namespace", namespace)

	var errs []error
	for _, obj := range Objects(namespace, p.cfg) {
		if err := p.createObject(ctx, obj); err != nil {
			errs = append(errs, fmt.Errorf("%s/%s: %w", obj.GetKind(), obj.GetName(), err))
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
```

**The worker logs the error and drops it** — `internal/watcher/watcher.go:221-230`

The only consumer of the error is the log line. The `inflight` marker is cleared but the `done` marker is not, and there is no requeue, backoff, or retry, so the failure is terminal for the process lifetime.

```go
func (w *Watcher) run(ctx context.Context, name string) {
	defer func() {
		w.queueMu.Lock()
		delete(w.inflight, name)
		w.queueMu.Unlock()
	}()
	if err := w.provisioner.Provision(ctx, name); err != nil {
		w.log.Error("failed to provision namespace", "namespace", name, "err", err)
	}
```

#### Dataflow

namespace Add event -\> Provision marks done -\> one of six creates fails -\> error logged only -\> IsDone true blocks all future attempts

- **Source:** transient Kubernetes API failure during create

- **Sink:** stale `done` map entry blocking re-provisioning

- **Outcome:** namespace permanently missing one or more default security resources

#### Reachability

No special attacker capability is needed to trigger the state (any API error suffices); deliberate triggering requires timing or a misconfigured default.

- **Attacker:** any actor able to induce a transient API failure, or an operator whose config makes one create invalid

- **Entry point:** namespace creation while an API error affects the create path

- **Outcome:** silently degraded security defaults for the namespace

#### Severity

**Medium** — Impact is medium: one or more security defaults are missing from the affected namespace (network isolation or resource containment, depending on which create failed), and the gap is invisible except in logs. Likelihood is medium: a single transient API failure during the six sequential creates is enough, and the create loop widens the failure window; a cluster API blip during backfill can hit many namespaces at once.

Raise to high if the cluster treats the NetworkPolicy as the primary tenant-isolation control and no other policy exists; lower to low if operators routinely run `--backfill` after failures or monitor the error log for re-provisioning.

#### Remediation

Mark the namespace done only after all six creates succeed; on failure, leave the name un-done (or move it to a failure set) and requeue it with backoff so the next event or a periodic reconcile retries provisioning.

Tests:
- Test: inject a create failure on the third object and assert the namespace is eventually fully provisioned after the API recovers.
- Test: assert a namespace whose NetworkPolicy create fails is re-attempted and ends with all six resources.

Preventive controls:
- Use a work queue with retry/backoff semantics (e.g. client-go workqueue) instead of a plain channel plus in-memory done map.
- Alert on the 'failed to provision namespace' log line so partial states are observable.

<a id="finding-3"></a>

### [3] Pre-created same-named resources masquerade as the watcher defaults (no ownership verification)

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | medium |
| Confidence rationale | The source proves the mechanism: fixed names (config.go:106-154), AlreadyExists treated as success with no read-back or label check (provisioner.go:110-119), and the `managed-by` label is never compared against existing objects (it appears only in config.go and templates.go, never in a GET or verification). Confidence is medium rather than high because exploitability depends on the tenant RBAC profile and relative API timing, both outside the repository. |
| Category | insecure_design |
| CWE | CWE-345 |
| Affected lines | internal/provisioner/provisioner.go:103-124, internal/config/config.go:106-137, internal/watcher/watcher.go:26 |

#### Summary

The watcher creates six resources with fixed names (production-quota, default-limits, myapp-sa, myapp-role, myapp-rolebinding, myapp-network-policy) and treats an AlreadyExists create error as success without verifying the pre-existing object is the one it would have created. The `app.kubernetes.io/managed-by` label is stamped onto the templates but never checked against existing objects. A namespace creator who also has create permission on the target resource type inside their namespace can race the watcher: create the namespace, then immediately create a permissive `production-quota` (or a lenient `myapp-network-policy`). The watcher's create returns 409, is treated as success, and the namespace is recorded done with the attacker's object as the effective default. The attacker can widen the race window arbitrarily by congesting the 1024-entry enqueue channel with other namespace creations.

#### Root Cause

The violated invariant is that the objects holding the fixed default names are the ones the watcher intends. Idempotency is implemented by trusting the error code: AlreadyExists is assumed to mean 'our object is already here', but the name is the only claim to the object and anyone with create permission in the namespace can hold it first. The `managed-by` label that would distinguish watcher-owned objects is stamped but never verified.

**AlreadyExists is treated as success with no ownership check** — `internal/provisioner/provisioner.go:110-119`

When the create returns AlreadyExists, the provisioner logs and returns success without reading the existing object back or comparing its labels/spec to the intended template. Whatever object holds the name is now accepted as the default.

```go
	_, err = p.client.Resource(mapping.Resource).Namespace(obj.GetNamespace()).
		Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			p.log.Debug("resource already exists, skipping",
				"kind", obj.GetKind(), "name", obj.GetName())
			return nil
		}
		return err
	}
```

**Resource names are fixed and predictable** — `internal/config/config.go:106-114`

All six names (default-limits, production-quota, myapp-sa, myapp-role, myapp-rolebinding, myapp-network-policy) are constants known to any tenant, so a same-name collision is trivially predictable.

```go
		LimitRange: LimitRangeDefaults{
			Name:                 "default-limits",
			DefaultMemory:        "256Mi",
			DefaultCPU:           "500m",
			DefaultRequestMemory: "128Mi",
			DefaultRequestCPU:    "100m",
			MaxMemory:            "2Gi",
			MaxCPU:               "2",
		},
```

**Bounded enqueue channel widens the race window** — `internal/watcher/watcher.go:26`

The namespace sits in a 1024-deep queue behind other creations; a tenant flooding namespace creates pushes their target namespace deep in the queue, giving the tenant seconds to minutes to win the name race.

```go
const defaultQueueSize = 1024
```

#### Validation

Verified: (1) names are fixed constants (config.go); (2) the create path treats AlreadyExists as success with no GET/label verification (provisioner.go:110-119); (3) the watcher's path from namespace Add event to create includes informer delivery plus queue latency (watcher.go:156-194), while the attacker's create is a direct single API call immediately after their namespace create returns, so the attacker's object typically lands first; (4) queue congestion (1024 entries) lets the attacker extend the delay arbitrarily. Grep confirms the managed-by label is never read back from existing objects anywhere.

Validation method: static source trace plus race-window analysis

**AlreadyExists is treated as success with no ownership check** — `internal/provisioner/provisioner.go:110-119`

When the create returns AlreadyExists, the provisioner logs and returns success without reading the existing object back or comparing its labels/spec to the intended template. Whatever object holds the name is now accepted as the default.

```go
	_, err = p.client.Resource(mapping.Resource).Namespace(obj.GetNamespace()).
		Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			p.log.Debug("resource already exists, skipping",
				"kind", obj.GetKind(), "name", obj.GetName())
			return nil
		}
		return err
	}
```

**Resource names are fixed and predictable** — `internal/config/config.go:106-114`

All six names (default-limits, production-quota, myapp-sa, myapp-role, myapp-rolebinding, myapp-network-policy) are constants known to any tenant, so a same-name collision is trivially predictable.

```go
		LimitRange: LimitRangeDefaults{
			Name:                 "default-limits",
			DefaultMemory:        "256Mi",
			DefaultCPU:           "500m",
			DefaultRequestMemory: "128Mi",
			DefaultRequestCPU:    "100m",
			MaxMemory:            "2Gi",
			MaxCPU:               "2",
		},
```

#### Dataflow

tenant creates namespace -\> tenant creates same-named resource -\> watcher create gets 409 -\> treated as success -\> done marked -\> attacker's object is the effective default

- **Source:** tenant Kubernetes API create calls (namespace, then same-named resource)

- **Sink:** AlreadyExists-as-success path in createObject

- **Outcome:** cluster's enforced default for the namespace is attacker-authored

#### Reachability

Requires only intra-namespace create permission plus timing; no privilege escalation or watcher compromise.

- **Attacker:** cluster tenant with create permission on the target resource type inside their namespace

- **Entry point:** Kubernetes API creates in the freshly created namespace

- **Outcome:** permissive attacker-authored default stands in for the enforced default

#### Severity

**Medium** — Impact is medium: the tenant replaces the enforced default for the chosen resource type with a self-authored permissive object, defeating the containment/isolation intent for their namespace while the operator believes defaults are applied. Likelihood is medium: it requires a tenant who can create the resource type inside their namespace (a common multi-tenant pattern) and winning a timing race, which the attacker can favor by loading the queue.

Raise to high for tenants who can create resourcequotas and networkpolicies but not update or delete them (the race is then their only escape and they cannot simply delete the watcher's objects); lower to low if tenants have no intra-namespace create permissions.

#### Remediation

On AlreadyExists, read the existing object back and verify it carries the `app.kubernetes.io/managed-by: namespace-watcher` label and the expected spec; if not, record a conflict (or adopt/overwrite per policy) instead of treating it as success.

Tests:
- Test: pre-create a same-named ResourceQuota without the managed-by label and assert the watcher does not mark the namespace provisioned (or reports the conflict).
- Test: pre-create a same-named object with the correct label and spec and assert it is accepted as the default.

Preventive controls:
- Select owned objects by label rather than by fixed name, so a tenant cannot squat the exact name the watcher would use.

<a id="finding-4"></a>

### [4] ClusterRole grants cluster-wide get/list on six resource types the code never uses

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Proven by exhaustive grep of non-test code: the only client calls in the provisioner are `Create` (provisioner.go:111) and in the watcher `Namespaces().List` (watcher.go:143). No `Get`, `List`, `Update`, or `Delete` on the six resource types exists anywhere outside tests. The ClusterRole text (deploy/namespace-watcher.yaml:35-44) explicitly grants the unused verbs. |
| Category | improper_privilege_management |
| CWE | CWE-250 |
| Affected lines | deploy/namespace-watcher.yaml:35-44, internal/provisioner/provisioner.go:103-124, README.md:78-79 |

#### Summary

The shipped ClusterRole grants `create, get, list` on limitranges, resourcequotas, serviceaccounts, roles, rolebindings, and networkpolicies in all namespaces, but the production code only ever calls `Create` on those types (provisioner.go:110-111) plus `Namespaces().List` at startup in new-only mode (watcher.go:143). The manifest comment 'read, for idempotency' is incorrect: idempotency comes from handling the AlreadyExists create error, not from reads. A compromised watcher ServiceAccount token therefore carries reconnaissance scope it does not need: enumerate and read every Role, RoleBinding, ServiceAccount, quota, and NetworkPolicy cluster-wide. This also contradicts the README's 'Least-privilege RBAC' claim.

#### Root Cause

The violated invariant is that the ServiceAccount's privileges match the operations the code actually performs. The ClusterRole was authored from an assumption that idempotency requires reads, but the implementation achieves idempotency by tolerating AlreadyExists on Create, leaving `get`/`list` as unexercised, unneeded scope on sensitive RBAC and quota objects cluster-wide.

**ClusterRole grants create, get, list cluster-wide** — `deploy/namespace-watcher.yaml:35-44`

The `get` and `list` verbs on all six resource types are granted cluster-wide. The code never issues a Get or List for any of these types, so the verbs are pure excess scope; the comment misattributes idempotency to reads when it actually comes from the AlreadyExists create-error path.

```yaml
  # Create (and read, for idempotency) the default resources in any namespace.
  - apiGroups: [""]
    resources: ["limitranges", "resourcequotas", "serviceaccounts"]
    verbs: ["create", "get", "list"]
  - apiGroups: ["rbac.authorization.k8s.io"]
    resources: ["roles", "rolebindings"]
    verbs: ["create", "get", "list"]
  - apiGroups: ["networking.k8s.io"]
    resources: ["networkpolicies"]
    verbs: ["create", "get", "list"]
```

**The provisioner only ever calls Create** — `internal/provisioner/provisioner.go:110-111`

This is the sole call the provisioner makes against the six resource types (confirmed by grep over all non-test Go code). The minimal grant needed is `create` plus (optionally) `get` if an ownership-verification fix is adopted.

```go
	_, err = p.client.Resource(mapping.Resource).Namespace(obj.GetNamespace()).
		Create(ctx, obj, metav1.CreateOptions{})
```

**README claims least-privilege RBAC** — `README.md:78-79`

The documentation asserts a least-privilege design and even says 'create/read', yet the code performs no reads at all, so the deployed grant exceeds both the code's needs and the documented intent.

```markdown
- **Least-privilege RBAC:** the watcher only needs to watch namespaces and
  create/read the six resource types it provisions.
```

#### Validation

Grepped all non-test Go files for Kubernetes client operations: the complete set is `Create` (provisioner.go:111) and `Namespaces().List` (watcher.go:143, used only in new-only mode and covered by the namespaces get/list/watch rule). No Get/List/Update/Delete on limitranges, resourcequotas, serviceaccounts, roles, rolebindings, or networkpolicies exists. The ClusterRole text grants exactly the unused get/list verbs on those types.

Validation method: exhaustive grep of client calls in non-test code against the granted RBAC

**ClusterRole grants create, get, list cluster-wide** — `deploy/namespace-watcher.yaml:35-44`

The `get` and `list` verbs on all six resource types are granted cluster-wide. The code never issues a Get or List for any of these types, so the verbs are pure excess scope; the comment misattributes idempotency to reads when it actually comes from the AlreadyExists create-error path.

```yaml
  # Create (and read, for idempotency) the default resources in any namespace.
  - apiGroups: [""]
    resources: ["limitranges", "resourcequotas", "serviceaccounts"]
    verbs: ["create", "get", "list"]
  - apiGroups: ["rbac.authorization.k8s.io"]
    resources: ["roles", "rolebindings"]
    verbs: ["create", "get", "list"]
  - apiGroups: ["networking.k8s.io"]
    resources: ["networkpolicies"]
    verbs: ["create", "get", "list"]
```

**The provisioner only ever calls Create** — `internal/provisioner/provisioner.go:110-111`

This is the sole call the provisioner makes against the six resource types (confirmed by grep over all non-test Go code). The minimal grant needed is `create` plus (optionally) `get` if an ownership-verification fix is adopted.

```go
	_, err = p.client.Resource(mapping.Resource).Namespace(obj.GetNamespace()).
		Create(ctx, obj, metav1.CreateOptions{})
```

#### Dataflow

compromised watcher SA token -\> cluster-wide get/list on RBAC/quota objects -\> reconnaissance of tenant privileges

- **Source:** stolen or leaked watcher ServiceAccount token

- **Sink:** API server get/list on roles, rolebindings, serviceaccounts, quotas, networkpolicies in all namespaces

- **Outcome:** cluster-wide RBAC and quota reconnaissance under the watcher identity

#### Reachability

Requires compromising the watcher workload or its token; the deployment's hardening (runAsNonRoot, dropped capabilities, read-only rootfs) narrows but does not remove that path.

- **Attacker:** in-cluster actor with access to the watcher's ServiceAccount token or process

- **Entry point:** watcher ServiceAccount credentials

- **Outcome:** excessive reconnaissance scope beyond what the watcher's code needs

#### Severity

**Low** — Impact is low-to-medium: the extra verbs enable cluster-wide RBAC and quota reconnaissance (useful for targeting privilege escalation) but no direct mutation beyond what `create` already allows. Likelihood is low: exploitation requires compromise of the watcher's SA token (container escape, token theft, or a bug in the watcher) or an operator deploying the manifest unchanged. The container hardening in the same manifest reduces (but does not eliminate) token-theft paths.

Raise to medium if the deployment environment makes SA-token compromise likely (shared nodes with weak isolation, permissive PodSecurity) or if the watcher gains untrusted-input handling over time.

#### Remediation

Remove the unused `get` and `list` verbs from the six resource types in the ClusterRole (keep `create`; add `get` only if an ownership-verification read is adopted) and correct the manifest comment and README to match the actual operations.

Tests:
- Test: run the full provisioning flow against a cluster where the ClusterRole grants only `create` on the six types plus get/list/watch on namespaces, and assert all six resources are created and idempotency holds.
- Test: assert the watcher fails closed (clear error) rather than silently succeeding if the RBAC grant is reduced further.

Preventive controls:
- Derive the ClusterRole from a static inventory of client calls (e.g. a lint or test that fails when a new verb appears in code without a matching RBAC entry).

<a id="finding-5"></a>

### [5] Default NetworkPolicy only constrains pods labeled app=myapp; typical workloads in new namespaces remain unrestricted

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Directly proven by the template source: podSelector matchLabels is `app: myapp` (config.go:138-140; templates.go:186-188), policyTypes are Ingress+Egress (config.go:141), and the DNS egress rule uses an empty namespaceSelector matching all namespaces (templates.go:165-179). The K8s semantics (policy applies only to selected pods; unselected pods are unrestricted) are standard and unchanged by the source. |
| Category | insecure_design |
| CWE | CWE-1188 |
| Affected lines | internal/provisioner/templates.go:181-211, internal/config/config.go:136-154, internal/provisioner/templates.go:165-179, README.md:20 |

#### Summary

The provisioned NetworkPolicy's podSelector matches only pods with the label `app: myapp` (config.go:138-140, applied at templates.go:186-188). Per Kubernetes NetworkPolicy semantics, a namespace's default isolation therefore applies only to workloads that opt in by carrying that label; any pod in a freshly created namespace without the label has no ingress or egress restriction at all. The README presents the policy as a blanket default ('Ingress only from the ingress controller; egress only to postgres + DNS'), which overstates its effect. Additionally, the AllowDNS rule (empty namespaceSelector, UDP/53 to all namespaces) gives labeled pods an unrestricted in-cluster DNS channel usable for exfiltration or tunneling through any in-cluster resolver.

#### Root Cause

The violated invariant is that a newly created namespace provides default network isolation for its workloads. The default policy is written for one specific application label (`app=myapp`) and is presented in the README as a general default; because Kubernetes NetworkPolicies only restrict the pods they select, every unlabeled workload in a new namespace is unrestricted by design, and the selected pods retain a cluster-wide DNS egress hole.

**Policy applies only to pods labeled app=myapp** — `internal/provisioner/templates.go:185-189`

`np.PodSelector` is `{"app": "myapp"}` (config.go:138-140), so the NetworkPolicy selects only pods carrying that exact label. Pods without it are outside the policy entirely and retain default-allow networking.

```go
		"spec": map[string]interface{}{
			"podSelector": map[string]interface{}{
				"matchLabels": stringMap(np.PodSelector),
			},
```

**The defaults encode the label gate** — `internal/config/config.go:136-154`

The default configuration is explicitly scoped to the `myapp` label and opens cluster-wide DNS egress; this is the value every provisioned namespace receives.

```go
		NetworkPolicy: NetworkPolicyDefaults{
			Name: "myapp-network-policy",
			PodSelector: map[string]string{
				"app": "myapp",
			},
			PolicyTypes: []string{"Ingress", "Egress"},
			Ingress: IngressRule{
				FromApp:  "nginx-ingress",
				Port:     3000,
				Protocol: "TCP",
			},
			Egress: EgressRule{
				ToApp:    "postgres",
				Port:     5432,
				Protocol: "TCP",
			},
			AllowDNS: true,
			DNSPort:  53,
			},
```

**DNS egress is allowed to every namespace** — `internal/provisioner/templates.go:165-179`

The empty namespaceSelector matches all namespaces, so labeled pods may send UDP/53 to any pod in any namespace. Combined with in-cluster DNS resolvers that forward upstream, this is a data-exfiltration and tunneling channel for selected pods.

```go
	if np.AllowDNS {
		egress = append(egress, map[string]interface{}{
			"to": []interface{}{
				map[string]interface{}{
					"namespaceSelector": map[string]interface{}{},
				},
			},
			"ports": []interface{}{
				map[string]interface{}{
					"protocol": "UDP",
					"port":     int64(np.DNSPort),
				},
			},
		})
	}
```

**README describes the policy as a blanket default** — `README.md:20`

The documentation omits the label precondition, so readers may assume every pod in a new namespace is constrained when in fact only `app=myapp` pods are.

```markdown
| `NetworkPolicy`| `myapp-network-policy` | Ingress only from the ingress controller; egress only to postgres + DNS |
```

#### Validation

Verified the template encodes podSelector matchLabels {app: myapp}, policyTypes \[Ingress, Egress\], a single ingress rule from app=nginx-ingress TCP/3000, a single egress rule to app=postgres TCP/5432, and (AllowDNS) UDP/53 to an empty namespaceSelector (all namespaces). Under standard K8s semantics, pods not matching the selector are not subject to the policy, so the 'default isolation' applies only to labeled pods.

Validation method: static review of the template against Kubernetes NetworkPolicy semantics

**Policy applies only to pods labeled app=myapp** — `internal/provisioner/templates.go:185-189`

`np.PodSelector` is `{"app": "myapp"}` (config.go:138-140), so the NetworkPolicy selects only pods carrying that exact label. Pods without it are outside the policy entirely and retain default-allow networking.

```go
		"spec": map[string]interface{}{
			"podSelector": map[string]interface{}{
				"matchLabels": stringMap(np.PodSelector),
			},
```

**The defaults encode the label gate** — `internal/config/config.go:136-154`

The default configuration is explicitly scoped to the `myapp` label and opens cluster-wide DNS egress; this is the value every provisioned namespace receives.

```go
		NetworkPolicy: NetworkPolicyDefaults{
			Name: "myapp-network-policy",
			PodSelector: map[string]string{
				"app": "myapp",
			},
			PolicyTypes: []string{"Ingress", "Egress"},
			Ingress: IngressRule{
				FromApp:  "nginx-ingress",
				Port:     3000,
				Protocol: "TCP",
			},
			Egress: EgressRule{
				ToApp:    "postgres",
				Port:     5432,
				Protocol: "TCP",
			},
			AllowDNS: true,
			DNSPort:  53,
			},
```

**DNS egress is allowed to every namespace** — `internal/provisioner/templates.go:165-179`

The empty namespaceSelector matches all namespaces, so labeled pods may send UDP/53 to any pod in any namespace. Combined with in-cluster DNS resolvers that forward upstream, this is a data-exfiltration and tunneling channel for selected pods.

```go
	if np.AllowDNS {
		egress = append(egress, map[string]interface{}{
			"to": []interface{}{
				map[string]interface{}{
					"namespaceSelector": map[string]interface{}{},
				},
			},
			"ports": []interface{}{
				map[string]interface{}{
					"protocol": "UDP",
					"port":     int64(np.DNSPort),
				},
			},
		})
	}
```

#### Dataflow

new namespace provisioned -\> pods without app=myapp label fall outside the policy -\> unrestricted networking

- **Source:** workload deployment in a newly created namespace

- **Sink:** network rules (or lack thereof) applied to the pod

- **Outcome:** default network isolation absent for typical (unlabeled) workloads

#### Reachability

No special capability is needed: any workload in a provisioned namespace that lacks the label is outside the policy.

- **Attacker:** tenant or workload author in any provisioned namespace

- **Entry point:** workload deployment (pod labels)

- **Outcome:** unrestricted ingress/egress despite the provisioned 'default' NetworkPolicy

#### Severity

**Low** — Impact is low: the tool's network-isolation default is weaker than documented, but it does apply (and is useful) to the labeled 'myapp' workload it targets, and tenants can opt in by labeling pods. The DNS egress rule creates a modest exfiltration channel for labeled pods. Likelihood is high in the sense that the condition holds by default for every unlabeled workload, but the security consequence depends on whether operators treat the policy as their isolation boundary.

Raise to medium if operators rely on this policy as the tenant-isolation control for arbitrary workloads rather than only app=myapp deployments.

#### Remediation

If namespace-level default isolation is the intent, select all pods (empty podSelector matchLabels) or make the label configurable and document the opt-in; constrain the DNS egress rule to the specific DNS server pods/namespace instead of an empty namespaceSelector.

Tests:
- Test: deploy an unlabeled pod in a provisioned namespace and assert its egress is restricted if isolation is the intended default (or assert and document that it is not).
- Test: assert a labeled pod cannot send UDP/53 to arbitrary namespaces once the DNS rule is scoped to the resolver.

Preventive controls:
- Align the README's per-resource table with the actual selector semantics so operators do not assume blanket isolation.

<a id="finding-6"></a>

### [6] Sustained namespace-creation bursts block the informer event loop and delay all default provisioning

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The blocking semantics are proven in source: the handler is registered synchronously (watcher.go:100-104) and blocks on a full channel (watcher.go:190-193). The claim that this stalls informer-wide event dispatch relies on standard client-go SharedInformer behavior (single dispatch goroutine per listener), which was not verified against a local copy of the client-go source in this offline scan; the drain-rate estimate likewise depends on cluster API latency. |
| Category | resource_exhaustion |
| CWE | CWE-400 |
| Affected lines | internal/watcher/watcher.go:190-193, internal/watcher/watcher.go:26, README.md:64-67 |

#### Summary

`onAdd` blocks on the bounded 1024-entry channel when it is full (`select { case w.queue <- name: case <-w.ctxDone(): }`), and client-go SharedInformer invokes event handlers synchronously in its dispatch loop. A sustained namespace-creation rate that outruns the 4 workers' drain rate therefore stalls all namespace event processing, contrary to the code and README claim that a burst 'never blocks the informer'. Consequence: security defaults for every subsequent namespace (including other tenants') are delayed for as long as the burst lasts, and the API-server watch buffer absorbs the backpressure.

#### Root Cause

The violated invariant is that namespace event processing stays responsive under load. The 'bounded queue' is bounded only in memory; its consumer is slow (six sequential API calls per namespace across 4 workers) and its producer is synchronous with the informer dispatch loop, so backpressure propagates into the watch-event pipeline instead of being absorbed by a requeue mechanism.

**onAdd blocks when the queue is full** — `internal/watcher/watcher.go:188-194`

With a full queue the select has no other ready case (the context is not done), so the handler goroutine blocks until a worker consumes an item. Because the informer dispatches handlers synchronously, this halts processing of all subsequent namespace events.

```go
	// Enqueue for a worker; blocks only if the queue is full, and drops only
	// on shutdown.
	select {
	case w.queue <- name:
	case <-w.ctxDone():
	}
```

**Queue capacity is 1024** — `internal/watcher/watcher.go:26`

The attacker needs only to keep more than 1024 namespaces in flight ahead of the drain rate to hold the informer stalled.

```go
const defaultQueueSize = 1024
```

**Documented claim that bursts never block the informer** — `README.md:64-67`

The documented invariant is false under a sustained rate that fills the bounded channel: the handler blocks, which by construction stalls the informer's event delivery for namespaces.

```markdown
- **client-go informer** on `v1/namespaces` for watching. Events are fanned out
  to a **bounded work queue** processed by a small pool of workers, so a burst
  of namespace creations never blocks the informer and work is deduplicated per
  namespace.
```

#### Validation

Verified the handler is registered as a plain `AddFunc` with no goroutine boundary (watcher.go:100-104), the enqueue select blocks on a full channel (watcher.go:190-193), and the queue holds at most 1024 names (watcher.go:26, 85). No metrics, drop policy, or requeue exists for items that cannot be enqueued. client-go SharedInformer delivers to handlers synchronously in the process loop, so a blocked handler stalls all pending events for that informer.

Validation method: static source trace of the enqueue path plus client-go dispatch semantics

**onAdd blocks when the queue is full** — `internal/watcher/watcher.go:188-194`

With a full queue the select has no other ready case (the context is not done), so the handler goroutine blocks until a worker consumes an item. Because the informer dispatches handlers synchronously, this halts processing of all subsequent namespace events.

```go
	// Enqueue for a worker; blocks only if the queue is full, and drops only
	// on shutdown.
	select {
	case w.queue <- name:
	case <-w.ctxDone():
	}
```

**Queue capacity is 1024** — `internal/watcher/watcher.go:26`

The attacker needs only to keep more than 1024 namespaces in flight ahead of the drain rate to hold the informer stalled.

```go
const defaultQueueSize = 1024
```

#### Dataflow

high-rate namespace creates -\> informer Add events -\> onAdd blocks on full queue -\> dispatch loop stalls -\> later namespaces' defaults delayed

- **Source:** scripted Kubernetes API namespace creates

- **Sink:** blocking channel send in onAdd

- **Outcome:** cluster-wide delay of security-default provisioning

#### Reachability

Requires only namespace-create permission and a sustained creation rate; no other capability.

- **Attacker:** cluster tenant with namespaces:create

- **Entry point:** Kubernetes API namespace creates

- **Outcome:** delayed (not lost) application of security defaults for all namespaces during the burst

#### Severity

**Low** — Impact is low-to-medium: the stall delays (does not lose) provisioning; once the burst ends the queue drains and defaults are applied. The attacker can also keep the watcher's CPU and the API-server watch channel busy. Likelihood is medium: it requires a sustained creation rate above the drain rate (roughly the 4 workers x six sequential API creates), which is achievable with a scripted client but not trivial.

Raise to medium if default-application latency is a security requirement (e.g. namespaces must be isolated before workloads run) and the cluster has no admission-time isolation.

#### Remediation

Make enqueue non-blocking: on a full queue, drop the event into a retry set and requeue via a periodic namespace re-list (or use a client-go workqueue with rate-limited retries), so the informer dispatch loop never blocks on worker drain rate.

Tests:
- Test: enqueue 2048 namespaces faster than the workers drain and assert informer event delivery (e.g. a control Add callback) continues without stalling.
- Test: assert every namespace created during a saturating burst is eventually provisioned.

Preventive controls:
- Expose queue depth and event-age metrics with alerts so sustained saturation is observable.
- Correct the README invariant to describe the actual backpressure behavior.

<a id="finding-7"></a>

### [7] Deployment uses a mutable :latest image tag for a highly privileged operator

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The mutable tag and pull policy are directly visible in deploy/namespace-watcher.yaml:85-86; the unpinned build base is visible in Dockerfile:4. The blast radius follows from the ClusterRole already established in this scan (create roles/rolebindings cluster-wide). |
| Category | supply_chain |
| CWE | CWE-494 |
| Affected lines | deploy/namespace-watcher.yaml:85-86, Dockerfile:4 |

#### Summary

The Deployment pins `image: ghcr.io/example/namespace-watcher:latest` with `imagePullPolicy: IfNotPresent`, and the Dockerfile build stage uses an unpinned `golang:1.22` base. The `:latest` tag is mutable: a registry compromise, a compromised release pipeline, or an accidental push can replace the binary, and the new image is picked up on the next pod scheduled onto a node without the cached image. Because the watcher's ServiceAccount holds a ClusterRole that can create Roles and RoleBindings in any namespace, a swapped image is a direct path to cluster-wide RBAC compromise. The container's own hardening (runAsNonRoot, dropped capabilities, read-only rootfs, seccomp RuntimeDefault, no host namespaces) is otherwise sound.

#### Root Cause

The violated invariant is that the privileged operator binary is immutable and reproducible. The deployment references a mutable tag instead of a content digest or versioned tag, and the build inputs are unpinned, so the artifact actually running under the powerful ClusterRole is not fixed by the manifest or the Dockerfile.

**Deployment pulls the mutable :latest tag** — `deploy/namespace-watcher.yaml:85-86`

`:latest` is a moving target in the registry. With IfNotPresent, any node that does not already cache the image will pull whatever `latest` currently points to on next scheduling, so a malicious or buggy push propagates to new pods and nodes without any manifest change.

```yaml
          image: ghcr.io/example/namespace-watcher:latest
          imagePullPolicy: IfNotPresent
```

**Build stage uses an unpinned base image** — `Dockerfile:4`

The build stage base is also a moving tag, so the toolchain itself is not reproducible; the runtime stage (distroless/static:nonroot) is likewise unpinned.

```dockerfile
FROM golang:1.22 AS build
```

#### Validation

Verified the image field is a `:latest` tag with IfNotPresent (yaml:85-86) and the Dockerfile uses unpinned `golang:1.22` and `gcr.io/distroless/static:nonroot` bases (Dockerfile:4, 18). Cross-referenced with the ClusterRole established earlier in this scan: the watcher SA can create roles and rolebindings in all namespaces, so a substituted image gains cluster-wide RBAC-authoring ability.

Validation method: static review of deployment manifest and Dockerfile

**Deployment pulls the mutable :latest tag** — `deploy/namespace-watcher.yaml:85-86`

`:latest` is a moving target in the registry. With IfNotPresent, any node that does not already cache the image will pull whatever `latest` currently points to on next scheduling, so a malicious or buggy push propagates to new pods and nodes without any manifest change.

```yaml
          image: ghcr.io/example/namespace-watcher:latest
          imagePullPolicy: IfNotPresent
```

**Build stage uses an unpinned base image** — `Dockerfile:4`

The build stage base is also a moving tag, so the toolchain itself is not reproducible; the runtime stage (distroless/static:nonroot) is likewise unpinned.

```dockerfile
FROM golang:1.22 AS build
```

#### Dataflow

registry push to :latest -\> node pulls new image on scheduling -\> malicious binary runs under watcher SA -\> creates RBAC for attacker

- **Source:** malicious image push to the registry tag

- **Sink:** process execution under the watcher ServiceAccount's ClusterRole

- **Outcome:** cluster-wide RBAC authoring under a trusted operator identity

#### Reachability

Requires compromise of the registry or release pipeline; no cluster access needed.

- **Attacker:** registry or CI/CD pipeline compromiser

- **Entry point:** image registry tag

- **Outcome:** arbitrary code execution with the watcher's cluster-wide create-roles privileges

#### Severity

**Low** — Impact would be high if realized (a malicious operator binary with cluster-wide create-roles rights), but likelihood is low: it requires a registry-level or release-pipeline compromise, and `IfNotPresent` means nodes that already cache the image keep running the old binary until rescheduled. The image reference is also an explicit example placeholder (ghcr.io/example/...), which lowers present-tense exposure but documents a pattern operators are likely to ship.

Raise to medium if the image is pulled from a registry the organization does not strictly sign and verify, or if imagePullPolicy is changed to Always.

#### Remediation

Pin the deployment image to an immutable digest (e.g. `@sha256:...`) or a unique versioned tag, pin Dockerfile base images by digest, and add image signature verification (e.g. cosign admission) so the privileged operator's binary identity is fixed and authentic.

Tests:
- Test: assert the Deployment's image field references a digest or immutable tag (manifest lint).
- Test: push a changed image under the same tag and assert the cluster does not run it (digest pinning / admission policy).

Preventive controls:
- Gate the privileged ClusterRole behind admission policies that require signed, digest-pinned images for the watcher namespace.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| main.go — CLI entrypoint, flag parsing, rest.Config construction | input-handling / configuration | No issue found | Flags (--kubeconfig, --backfill) are passed to config.Load; no attacker-controlled data is interpreted unsafely. The missing --backfill default is reported against the deployment manifest rather than here. |
| internal/watcher — SharedInformer handler, bounded queue, known/inflight/done state | resource-exhaustion / state-machine correctness | Reported | Blocking enqueue under burst load (informer-event-stall) and the AddFunc-only handler (no delete/Update handler) are reported; the AddFunc-only design is the enabler for the namespace-recreate escape. |
| internal/provisioner — resource build, create loop, AlreadyExists handling, NetworkPolicy builder | insecure-design / privilege | Reported | Core defect surface: done-map never cleared (namespace-recreate-escape), no retry on partial failure (provision-failure-no-retry), AlreadyExists treated as success without spec verification (default-name-spoofing), and the podSelector label gate (networkpolicy-label-gate). |
| internal/config — fixed resource names and NetworkPolicy defaults | insecure-design | Reported | Fixed, predictable resource names are cited by default-name-spoofing; the podSelector app=myapp label and DNS egress rules are cited by networkpolicy-label-gate. The defaults themselves are reasonable; the defects are in how the provisioner consumes them. |
| deploy/namespace-watcher.yaml — ServiceAccount, ClusterRole, Deployment | privilege-management / supply-chain / availability | Reported | ClusterRole grants get/list on six resource types the code never exercises (clusterrole-unused-verbs); the Deployment pins :latest with IfNotPresent (mutable-image-tag); the absent --backfill flag keeps the recreate escape live across restarts. |
| Dockerfile — multi-stage build and base image | supply-chain | Reported | Unpinned golang:1.22 base (no digest) is cited by mutable-image-tag as part of the image-supply-chain risk. |
| Makefile — build, test, vet, and image targets | build-system | No issue found | No secret exfiltration, remote-code execution, or unsafe remote fetch in the targets. |
| go.mod / go.sum — Go module dependency set | supply-chain | No issue found | Standard k8s.io client-go and transitive dependencies; no suspicious or typosquatted module, no GOFLAGS/vendor tampering. Dependency currency against the latest patched releases is recorded as an open question (offline). |
| watcher_test.go / provisioner_test.go / config_test.go | test-integrity | No issue found | Tests do not weaken security controls or embed credentials; they corroborate the reviewed behavior (queue, done-map, create ordering). |
| README.md — documented security behavior and threat model | assurance / documentation | Reported | The README asserts idempotent, prompt default provisioning and least-privilege RBAC; the recreate escape and unused-verbs findings show the implemented behavior diverges from these documented guarantees. |
| bin/namespace-watcher — committed compiled binary | supply-chain | Not applicable | Compiled artifact excluded from static source analysis. Whether it corresponds to the audited source is an open question. |

## Open Questions And Follow Up

- What is the typical tenant RBAC profile? Exploitability of the namespace-recreate escape and default-name spoofing depends on whether a tenant can delete namespaces and/or create ResourceQuota, LimitRange, NetworkPolicy, Role, and RoleBinding. If tenants are read-only within namespaces and cannot delete namespaces, the practical impact is reduced.
  - Follow-up prompt: Provide a representative tenant ServiceAccount + Role/ClusterRole (or Pod Security admission configuration) so the exploit preconditions can be confirmed.
- Are there compensating cluster-level controls outside this repository (admission webhooks, an admission controller that re-applies per-namespace defaults, cluster ResourceQuota, Pod Security enforcement) that would re-establish or independently enforce the defaults this watcher is responsible for?
  - Follow-up prompt: List any admission policies or cluster-scoped quotas that act as a backstop for per-namespace defaults.
- The pinned dependency set (k8s.io/\* v0.31.3, golang.org/x/net v0.26.0, golang.org/x/sys v0.21.0, golang.org/x/oauth2 v0.21.0, Go 1.22.2) predates later security-relevant releases. Offline analysis could not verify current CVE status. Confirm dependency CVE posture from a connected SCA source.
  - Follow-up prompt: Run a connected SCA/OSV scan against go.mod and report any reachable CVEs.
- The committed bin/namespace-watcher binary was not verified to correspond to the current source. Confirm it is built from the audited source or remove it from the repository.
  - Follow-up prompt: Confirm the binary's build provenance or drop it from version control.
