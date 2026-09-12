# Security Review: namespace-watch-dv4f

## Scope

Repository-wide static security scan of the Go Kubernetes namespace-watcher operator at /sandbox/deepseekv4flash/namespace-watch-dv4f. All 11 project files (9 source files, go.mod, go.sum) were reviewed. The controller watches Namespace creation events and provisions default LimitRange, ResourceQuota, ServiceAccount, Role, RoleBinding and NetworkPolicy objects into each new namespace via the dynamic client. Analysis was static source review only; no runtime cluster execution or network access was performed.

- Scan mode: repository
- Target kind: directory_snapshot
- Target ID: codex-security-target/v1:sha256:5db866c350dd6faa6c37d46d46aa20960991a6a3a6e446d83cd4bf6a21bc9a50
- Snapshot digest: codex-security-snapshot/v1:sha256:080c16024ca95965dee057c17c6a6c6337ad26a96bfb32e344e68687e9d2440d
- Inventory strategy: directory
- Included paths: ., main.go, go.mod, internal/config/config.go, internal/controller/controller.go, internal/resources/provisioner.go, internal/resources/registry.go, internal/resources/limitrange.go, internal/resources/resourcequota.go, internal/resources/rbac.go, internal/resources/networkpolicy.go
- Excluded paths: none
- Runtime or test status: not recorded

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 3 |
| Severity mix | medium: 1, low: 2 |
| Confidence mix | high: 1, medium: 2 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

A cluster-level Kubernetes operator watches Namespace creation events and auto-provisions a fixed set of default resources (LimitRange, ResourceQuota, a deny-by-default NetworkPolicy, and an application ServiceAccount/Role/RoleBinding) into every non-excluded namespace using the dynamic client. The security objectives are baseline isolation and resource-limit enforcement for newly created namespaces.

### Assets

- auto-provisioned default-deny NetworkPolicy (myapp-network-policy)
- auto-provisioned ResourceQuota (production-quota) limiting CPU/memory/pods/services/PVCs
- auto-provisioned ServiceAccount myapp-sa and its Role/RoleBinding
- cluster stability under resource limits

### Trust Boundaries

- Kubernetes API server (source of Namespace create events) to controller reconcile loop
- controller dynamic-client writes to every non-excluded namespace
- namespace tenants with RBAC edit rights over resources within their own namespace

### Attacker Capabilities

- A principal able to create namespaces that trigger provisioning
- A tenant or workload holding write (delete/update) access to resources inside its own provisioned namespace
- A workload running the auto-created myapp-sa ServiceAccount token

### Security Objectives

- Deny-by-default network isolation stays enforced for a namespace's lifetime
- Resource usage remains bounded by the default ResourceQuota
- Default identities follow least privilege and do not expose sensitive data unnecessarily

### Assumptions

- Provisioned namespaces may be populated by tenants who are less trusted than cluster administrators
- The controller's own ServiceAccount is granted cluster-wide create rights for the provisioned resource kinds (no deployment manifest is present in-scope to verify least privilege)

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Baseline security controls (default-deny NetworkPolicy and ResourceQuota) are created once and never reconciled, so a namespace tenant can permanently disable them](#finding-1) | medium | high | inline below |
| [NetworkPolicy egress rule uses an empty namespaceSelector for DNS, permitting outbound UDP/53 to arbitrary destinations and enabling DNS-tunnel exfiltration from myapp pods](#finding-2) | low | medium | inline below |
| [Every new namespace auto-receives a ServiceAccount bound to a Role granting get/list over all ConfigMaps, expanding the read surface without tenant opt-in](#finding-3) | low | medium | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Baseline security controls (default-deny NetworkPolicy and ResourceQuota) are created once and never reconciled, so a namespace tenant can permanently disable them

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Confirmed directly in source: applyOne (provisioner.go) has no update/delete path, controller.go registers only AddFunc and UpdateFunc (UpdateFunc is a no-op), the informer watches Namespace objects only, and no ownerReference is set on any provisioned object. |
| Category | Protection Mechanism Failure |
| CWE | CWE-693 |
| Affected lines | internal/resources/provisioner.go:43-58, internal/controller/controller.go:121-128, internal/controller/controller.go:45-48, internal/resources/networkpolicy.go:20-82 |

#### Summary

The controller provisions a deny-by-default NetworkPolicy and a ResourceQuota into each new namespace only at creation time. `applyOne` creates an object and silently returns on `AlreadyExists`, `handleUpdate` ignores every update event, and no `DeleteFunc` is registered on the namespace informer. Once provisioned, deletion or weakening of these security controls inside a namespace is never detected or restored.

#### Root Cause

The violated invariant is that baseline security controls must remain present and enforced for the lifetime of a namespace. The create-once reconcile loop violates this: deletion or weakening of the NetworkPolicy/ResourceQuota produces no Namespace event, `handleUpdate` discards updates, and `applyOne` never restores an absent object.

**Informers register no DeleteFunc** — `internal/controller/controller.go:45-48`

Only Add and Update handlers are registered; there is no DeleteFunc, so deletion of a provisioned control never re-queues its namespace.

```go
namespaceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
	AddFunc:    c.handleAdd,
	UpdateFunc: c.handleUpdate,
})
```

**Update events are ignored** — `internal/controller/controller.go:121-128`

Modifications to a namespace's provisioned controls never re-enter the work queue, so weakened or deleted controls are never reconciled.

```go
func (c *Controller) handleUpdate(oldObj, newObj interface{}) {
	// We provision only on creation; updates to existing namespaces are ignored
	_ = oldObj ...
}
```

**applyOne creates and never updates/deletes** — `internal/resources/provisioner.go:43-58`

After the initial Create, any AlreadyExists (or deleted-then-ignored) state is tolerated; there is no Get/Update or restore path for the protected controls.

```go
if _, err := p.dynamicClient.Resource(res.GVR).Namespace(namespace).
	Create(ctx, obj, metav1.CreateOptions{}); err != nil {
	if apierrors.IsAlreadyExists(err) {
		klog.V(2).Infof("skipping %s %q/%q: already exists", ...)
		return nil
	}
	return err
}
```

#### Validation

Source confirms that no code path recreates or reverts a deleted/weakened NetworkPolicy or ResourceQuota: the informer watches only Namespace objects, `handleUpdate` is a no-op, no DeleteFunc is registered, and `applyOne` returns nil on AlreadyExists without an update branch. No ownerReference is set anywhere.

Validation method: static source trace

**Informers register no DeleteFunc** — `internal/controller/controller.go:45-48`

Only Add and Update handlers are registered; there is no DeleteFunc, so deletion of a provisioned control never re-queues its namespace.

```go
namespaceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
	AddFunc:    c.handleAdd,
	UpdateFunc: c.handleUpdate,
})
```

**Update events are ignored** — `internal/controller/controller.go:121-128`

Modifications to a namespace's provisioned controls never re-enter the work queue, so weakened or deleted controls are never reconciled.

```go
func (c *Controller) handleUpdate(oldObj, newObj interface{}) {
	// We provision only on creation; updates to existing namespaces are ignored
	_ = oldObj ...
}
```

**applyOne creates and never updates/deletes** — `internal/resources/provisioner.go:43-58`

After the initial Create, any AlreadyExists (or deleted-then-ignored) state is tolerated; there is no Get/Update or restore path for the protected controls.

```go
if _, err := p.dynamicClient.Resource(res.GVR).Namespace(namespace).
	Create(ctx, obj, metav1.CreateOptions{}); err != nil {
	if apierrors.IsAlreadyExists(err) {
		klog.V(2).Infof("skipping %s %q/%q: already exists", ...)
		return nil
	}
	return err
}
```

#### Dataflow

namespace create event -\> handleAdd -\> work queue -\> syncNamespace -\> provisioner.Apply -\> applyOne Create; later control deletion has no path back into the queue

- **Source:** a tenant's delete/update action on a provisioned resource inside its own namespace

- **Sink:** absence of any reconcile/restore branch in the controller

- **Outcome:** default-deny isolation and resource limits are silently lost for the namespace lifetime

**Update events are ignored** — `internal/controller/controller.go:121-128`

Modifications to a namespace's provisioned controls never re-enter the work queue, so weakened or deleted controls are never reconciled.

```go
func (c *Controller) handleUpdate(oldObj, newObj interface{}) {
	// We provision only on creation; updates to existing namespaces are ignored
	_ = oldObj ...
}
```

**applyOne creates and never updates/deletes** — `internal/resources/provisioner.go:43-58`

After the initial Create, any AlreadyExists (or deleted-then-ignored) state is tolerated; there is no Get/Update or restore path for the protected controls.

```go
if _, err := p.dynamicClient.Resource(res.GVR).Namespace(namespace).
	Create(ctx, obj, metav1.CreateOptions{}); err != nil {
	if apierrors.IsAlreadyExists(err) {
		klog.V(2).Infof("skipping %s %q/%q: already exists", ...)
		return nil
	}
	return err
}
```

#### Reachability

The attacker must hold write (delete/update) RBAC on at least the NetworkPolicy and ResourceQuota resources inside a provisioned namespace, which is common for tenant developers in their own namespace.

- **Attacker:** namespace tenant or workload with local edit rights

- **Entry point:** delete/weaken myapp-network-policy or production-quota

- **Outcome:** security controls permanently disabled

#### Severity

**Medium** — High impact (loss of the operator's baseline isolation and resource-limit enforcement, which can enable unbounded cluster resource consumption) with high likelihood for a tenant who already holds RBAC write rights in their own namespace. Rated medium rather than high because this is an explicit, documented design choice ('we do not fight with operators over resource changes') and the operator is provisioning convenience defaults rather than enforcing policy against a hostile namespace, so the enforcement invariant is only implicit. Would be raised to high if namespaces are explicitly treated as untrusted tenants.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Adopt desired-state reconciliation: watch the provisioned resource kinds (or use server-side apply with an owner reference/label), and restore or revert the default-deny NetworkPolicy, ResourceQuota, LimitRange, Role/RoleBinding when they are deleted or drift; alternatively enforce these controls via a tamper-resistant admission policy engine rather than create-once provisioning.

Tests:
- Assert that deleting myapp-network-policy triggers re-creation of the deny-by-default policy.
- Assert that modifying production-quota to remove limits is reverted on next reconcile.

Preventive controls:
- Set ownerReferences on provisioned objects so deletion/garbage-collection and namespace lifecycle are coupled.
- Scope enforcement resources with an operator-owned label/annotation so only operator-managed objects are reconciled.

<a id="finding-2"></a>

### [2] NetworkPolicy egress rule uses an empty namespaceSelector for DNS, permitting outbound UDP/53 to arbitrary destinations and enabling DNS-tunnel exfiltration from myapp pods

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The empty namespaceSelector semantics are confirmed by Kubernetes NetworkPolicy behavior and source, but actual exfiltration impact depends on workloads and cluster DNS topology (an ipBlock target was not specified). |
| Category | Improper Authorization |
| CWE | CWE-285 |
| Affected lines | internal/resources/networkpolicy.go:66-78, internal/resources/networkpolicy.go:32-50 |

#### Summary

The default NetworkPolicy declares deny-by-default Egress but includes a DNS rule whose `to` is only `namespaceSelector: {}`. An empty namespaceSelector matches every pod in every namespace, so any compromised `app: myapp` pod can send UDP/53 traffic to arbitrary hosts (DNS tunneling) rather than just cluster DNS.

#### Root Cause

The violated invariant is that egress should be deny-by-default, allowing DNS only to cluster-resolver endpoints. The empty `namespaceSelector` broadens the permitted DNS destination to arbitrary hosts/pods cluster-wide, opening a data-exfiltration channel for matched workloads.

**DNS egress matches all namespaces via empty namespaceSelector** — `internal/resources/networkpolicy.go:66-71`

An empty namespaceSelector selects every namespace; with no podSelector/IPBlock in this rule, it permits UDP/53 egress to any host, not only cluster DNS.

```go
"to": []interface{}{
	map[string]interface{}{
		"namespaceSelector": map[string]interface{}{},
	},
},
```

**Egress port UDP 53** — `internal/resources/networkpolicy.go:72-78`

The permissive destination is combined with UDP/53, the port classically used for DNS tunneling.

```go
"ports": []interface{}{
	map[string]interface{}{"protocol": "UDP", "port": int64(53)},
},
```

#### Validation

Source shows the policy sets Egress in policyTypes and then grants UDP/53 egress with a destination of only an empty namespaceSelector, which Kubernetes interprets as all namespaces. Counterevidence: DNS egress to cluster DNS is generally required, and the rule is port-restricted to UDP/53.

Validation method: static source trace

**DNS egress matches all namespaces via empty namespaceSelector** — `internal/resources/networkpolicy.go:66-71`

An empty namespaceSelector selects every namespace; with no podSelector/IPBlock in this rule, it permits UDP/53 egress to any host, not only cluster DNS.

```go
"to": []interface{}{
	map[string]interface{}{
		"namespaceSelector": map[string]interface{}{},
	},
},
```

**Egress port UDP 53** — `internal/resources/networkpolicy.go:72-78`

The permissive destination is combined with UDP/53, the port classically used for DNS tunneling.

```go
"ports": []interface{}{
	map[string]interface{}{"protocol": "UDP", "port": int64(53)},
},
```

#### Dataflow

compromised myapp pod -\> outbound UDP/53 -\> empty namespaceSelector matches all destinations -\> arbitrary external host (DNS tunneling)

- **Source:** network egress from a compromised app:myapp pod

- **Sink:** arbitrary UDP/53 destination permitted by the policy

- **Outcome:** data exfiltration channel outside the intended egress allowlist

**DNS egress matches all namespaces via empty namespaceSelector** — `internal/resources/networkpolicy.go:66-71`

An empty namespaceSelector selects every namespace; with no podSelector/IPBlock in this rule, it permits UDP/53 egress to any host, not only cluster DNS.

```go
"to": []interface{}{
	map[string]interface{}{
		"namespaceSelector": map[string]interface{}{},
	},
},
```

**Egress port UDP 53** — `internal/resources/networkpolicy.go:72-78`

The permissive destination is combined with UDP/53, the port classically used for DNS tunneling.

```go
"ports": []interface{}{
	map[string]interface{}{"protocol": "UDP", "port": int64(53)},
},
```

#### Reachability

Requires prior compromise of (or malicious code running as) a pod matching the policy's `app: myapp` podSelector, plus network reachability to an external DNS service.

- **Attacker:** compromised app:myapp workload

- **Entry point:** outbound UDP/53 packets from the compromised pod

- **Outcome:** DNS-tunnel exfiltration to arbitrary hosts

#### Severity

**Low** — Impact is limited to workloads already matched by the policy's `app: myapp` podSelector and requires a prior compromise of such a workload; it weakens, but does not eliminate, the deny-by-default egress posture. Allowing DNS egress is a conventional pattern.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Scope the DNS egress rule to cluster DNS: add an ipBlock covering the kube-dns/coreDNS service CIDR (and, if supported, a namespaceSelector targeting kube-system plus a label selecting the DNS pods) instead of an empty namespaceSelector that matches every namespace.

Tests:
- Assert the policy's DNS egress rule targets only the cluster resolver IP range.
- Assert no UDP/53 egress is allowed to arbitrary namespaceSelectors.

Preventive controls:
- Document and pin the cluster DNS service IP/CIDR in a configurable constant.
- Prefer explicit ipBlock allowlists for DNS/egress over blank namespace selectors.

<a id="finding-3"></a>

### [3] Every new namespace auto-receives a ServiceAccount bound to a Role granting get/list over all ConfigMaps, expanding the read surface without tenant opt-in

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | The Role/RoleBinding/ServiceAccount creation in every namespace is confirmed in source, but whether ConfigMaps actually hold sensitive data and whether the default SA is used by attacker-influenceable workloads depends on cluster deployment, which is not present in-scope. |
| Category | Improper Privilege Management |
| CWE | CWE-269 |
| Affected lines | internal/resources/rbac.go:47-63, internal/resources/registry.go:30-36 |

#### Summary

The registry unconditionally provisions into every new namespace a fixed ServiceAccount `myapp-sa`, a Role granting `get`/`list` on the core `configmaps` resource across the whole namespace, and a RoleBinding binding them. Any workload that runs with this default SA can read all ConfigMaps in the namespace, including ones holding sensitive values.

#### Root Cause

The violated invariant is least privilege on automatically-created identities. A fixed-name SA is auto-bound to a namespace-wide ConfigMap read role in every provisioned namespace regardless of need, granting broad read access to ConfigMaps (which may hold sensitive values) to anything able to run as that SA.

**Role grants get/list on all ConfigMaps** — `internal/resources/rbac.go:47-63`

The Role grants read access to every ConfigMap in the namespace with no resource-name restriction.

```go
"rules": []interface{}{
	map[string]interface{}{
		"apiGroups": []interface{}{""},
		"resources": []interface{}{"configmaps"},
		"verbs":     []interface{}{"get", "list"},
	},
},
```

**RoleBinding binds myapp-sa to the Role in each namespace** — `internal/resources/rbac.go:66-87`

The fixed SA is bound to the ConfigMap-read Role automatically; every provisioned namespace gets this identity.

```go
"subjects": []interface{}{
	map[string]interface{}{"kind": "ServiceAccount", "name": ServiceAccountName, "namespace": namespace},
},
"roleRef": map[string]interface{}{"kind": "Role", "apiGroup": "rbac.authorization.k8s.io", "name": RoleName},
```

**SA/Role/RoleBinding registered unconditionally** — `internal/resources/registry.go:30-36`

The three RBAC builders run for every new namespace with no configuration gate, so the read grant is applied even where unused.

```go
{GVR: ServiceAccountGVR, Build: BuildServiceAccount},
{GVR: RoleGVR, Build: BuildRole},
{GVR: RoleBindingGVR, Build: BuildRoleBinding},
```

#### Validation

Source confirms the Role, ServiceAccount and RoleBinding are built and registered for every namespace (rbac.go, registry.go) with read-only get/list on all configmaps; the grant is namespace-scoped and excludes Secrets.

Validation method: static source trace

**Role grants get/list on all ConfigMaps** — `internal/resources/rbac.go:47-63`

The Role grants read access to every ConfigMap in the namespace with no resource-name restriction.

```go
"rules": []interface{}{
	map[string]interface{}{
		"apiGroups": []interface{}{""},
		"resources": []interface{}{"configmaps"},
		"verbs":     []interface{}{"get", "list"},
	},
},
```

**RoleBinding binds myapp-sa to the Role in each namespace** — `internal/resources/rbac.go:66-87`

The fixed SA is bound to the ConfigMap-read Role automatically; every provisioned namespace gets this identity.

```go
"subjects": []interface{}{
	map[string]interface{}{"kind": "ServiceAccount", "name": ServiceAccountName, "namespace": namespace},
},
"roleRef": map[string]interface{}{"kind": "Role", "apiGroup": "rbac.authorization.k8s.io", "name": RoleName},
```

**SA/Role/RoleBinding registered unconditionally** — `internal/resources/registry.go:30-36`

The three RBAC builders run for every new namespace with no configuration gate, so the read grant is applied even where unused.

```go
{GVR: ServiceAccountGVR, Build: BuildServiceAccount},
{GVR: RoleGVR, Build: BuildRole},
{GVR: RoleBindingGVR, Build: BuildRoleBinding},
```

#### Dataflow

namespace create -\> registry -\> BuildRole/BuildRoleBinding -\> role granting get/list configmaps -\> any pod running as myapp-sa reads all ConfigMaps

- **Source:** workload or tenant using the auto-provisioned myapp-sa identity

- **Sink:** ConfigMap list/get authorization granted to that identity

- **Outcome:** unnecessary read access to sensitive ConfigMap data in the namespace

**Role grants get/list on all ConfigMaps** — `internal/resources/rbac.go:47-63`

The Role grants read access to every ConfigMap in the namespace with no resource-name restriction.

```go
"rules": []interface{}{
	map[string]interface{}{
		"apiGroups": []interface{}{""},
		"resources": []interface{}{"configmaps"},
		"verbs":     []interface{}{"get", "list"},
	},
},
```

**RoleBinding binds myapp-sa to the Role in each namespace** — `internal/resources/rbac.go:66-87`

The fixed SA is bound to the ConfigMap-read Role automatically; every provisioned namespace gets this identity.

```go
"subjects": []interface{}{
	map[string]interface{}{"kind": "ServiceAccount", "name": ServiceAccountName, "namespace": namespace},
},
"roleRef": map[string]interface{}{"kind": "Role", "apiGroup": "rbac.authorization.k8s.io", "name": RoleName},
```

#### Reachability

The attacker must be able to schedule a pod using myapp-sa or otherwise act as it, i.e. already have some authority within the same namespace; the grant expands that authority to all ConfigMaps.

- **Attacker:** workload/tenant acting as myapp-sa

- **Entry point:** running a pod with the default SA token in a provisioned namespace

- **Outcome:** read access to all ConfigMaps in the namespace

#### Severity

**Low** — The grant is namespace-scoped, read-only, and does not cover Secrets; exploiting it requires already being able to run a workload as `myapp-sa` in the same namespace, so the boundary crossed is limited. Rated low (least-privilege hardening gap) rather than medium because no cross-trust-boundary privilege gain is established from source.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Do not auto-provision myapp-sa/Role/RoleBinding for every namespace, or gate it behind configuration so tenants opt in; if kept, restrict the Role to a labeled resource-name subset and document that ConfigMaps may hold sensitive data.

Tests:
- Assert that namespaces provisioned without the feature enabled receive no default RoleBinding.
- Assert that myapp-sa cannot list ConfigMaps outside the allowed resource-name set.

Preventive controls:
- Provision the RBAC objects only when a namespace opts in via an annotation/label.
- Prefer scoped resourceNames rules over blanket configmap get/list.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| Namespace watch loop and work-queue reconciliation (controller.go) | Protection-mechanism enforcement; create-once provisioning is never reconciled after deletion/drift | Reported | handleAdd/handleUpdate/syncNamespace reviewed; handleUpdate is a no-op and no DeleteFunc is registered. Finding: protection-mechanism.create-once-no-reconcile. |
| Provisioner dynamic-client create path (provisioner.go, registry.go) | Create-only apply with no update/restore branch; aggregated error handling | Reported | applyOne reviewed in full. Finding: protection-mechanism.create-once-no-reconcile. |
| Auto-provisioned ServiceAccount/Role/RoleBinding (rbac.go) | Least-privilege on automatically created identities | Reported | Fixed myapp-sa bound to a namespace-wide ConfigMap get/list role in every new namespace. Finding: privilege-management.default-configmap-role. |
| Default NetworkPolicy definition (networkpolicy.go) | Deny-by-default ingress/egress posture; DNS egress scoping | Reported | Egress DNS rule uses an empty namespaceSelector. Findings: protection-mechanism.create-once-no-reconcile, access-control.broad-dns-egress. |
| LimitRange and ResourceQuota defaults (limitrange.go, resourcequota.go) | Resource-limit enforcement lifecycle | Reported | Static default values reviewed; quotas are enforced only at creation time. Finding: protection-mechanism.create-once-no-reconcile. |
| Client configuration bootstrap (config.go, main.go) | Credential handling / client construction | No issue found | Uses in-cluster or kubeconfig credential machinery; no hardcoded credentials or secret storage found. |
| Dependency manifest (go.mod, go.sum) | Third-party dependency versions | No issue found | No known-vulnerability version matching was performed in-scope; recorded as an open question rather than a confirmed issue. |

## Open Questions And Follow Up

- Are the pinned third-party dependency versions (go.mod / go.sum) affected by any known published CVEs?
  - Follow-up prompt: Run a dependency vulnerability scan against k8s.io/client-go, k8s.io/api and transitive dependencies listed in go.mod.
- What RBAC ClusterRole/ClusterRoleBinding is actually deployed for the operator's own ServiceAccount, and is it scoped to least privilege?
  - Follow-up prompt: Locate the deployment manifest (not present in-scope) and verify the operator runs with a restricted ClusterRole covering only limitranges, resourcequotas, serviceaccounts, roles, rolebindings, networkpolicies and namespaces.
- Are provisioned namespaces treated as untrusted tenants where default-deny NetworkPolicy and ResourceQuota must be enforced against tampering?
  - Follow-up prompt: Confirm the intended trust model to decide whether create-once provisioning (CWE-693) should be rated medium or high.
