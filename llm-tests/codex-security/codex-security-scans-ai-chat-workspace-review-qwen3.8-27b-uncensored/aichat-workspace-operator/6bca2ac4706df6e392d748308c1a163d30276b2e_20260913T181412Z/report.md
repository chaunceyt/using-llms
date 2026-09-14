# Security Review: aichat-workspace-operator

## Scope

Repository-wide standard security scan of the aichat-workspace-operator Kubernetes operator (Go, kubebuilder) including CRD API, controller reconcile logic, k8s/ollama adapters, shipped kustomize/hack manifests, RBAC, Dockerfile, Makefile, CI workflows, and tests. All 386 repository files were inventoried; all product source was reviewed.

- Scan mode: repository
- Target kind: git_revision
- Target ID: codex-security-target/v1:sha256:3a475a0906b184c3994aa0235ad59efbc7f08af55880217f6711047c8f439f83
- Revision: 6bca2ac4706df6e392d748308c1a163d30276b2e
- Inventory strategy: repository
- Included paths: .
- Excluded paths: none
- Runtime or test status: not recorded

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 5 |
| Severity mix | high: 1, medium: 2, low: 2 |
| Confidence mix | high: 4, medium: 1 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

No explicit canonical threat-model summary was recorded.

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Unvalidated workspaceName allows deleting arbitrary namespaces and injecting operator-managed resources into any namespace](#finding-1) | high | high | inline below |
| [Cross-tenant workspace-name collision exposes another tenant's unauthenticated Open WebUI](#finding-2) | medium | medium | inline below |
| [Shipped manifests embed weak, hardcoded database credentials](#finding-3) | medium | high | inline below |
| [Open WebUI workload is deployed as root without a security context](#finding-4) | low | high | inline below |
| [Manager exposes unauthenticated pprof/expvar/statsviz debug endpoints and development logging](#finding-5) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Unvalidated workspaceName allows deleting arbitrary namespaces and injecting operator-managed resources into any namespace

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | high |
| Confidence rationale | Direct source trace: the CRD Go type and generated CRD schema contain no validation beyond immutability; `deleteAIChatWorkspace` deletes the namespace named by `spec.WorkspaceName` with no ownership or reserved-name check; `handleReconcile` targets every resource at `Spec.WorkspaceName`; the shipped ClusterRole grants wildcard verbs on `namespaces`, `resourcequotas`, and related core resources cluster-wide. No live-cluster reproduction was performed. |
| Category | Input Validation |
| CWE | CWE-20, CWE-269 |
| Affected lines | internal/controller/aichatworkspace_controller.go:232-248, api/v1alpha1/aichatworkspace_types.go:25-28, internal/controller/delete.go:35-49, internal/controller/handle_reconcile.go:44-55, internal/controller/namespace.go:41-49, internal/adapters/k8s/common.go:169-176, internal/constants/constants.go:51-54, config/rbac/role.yaml:13-24 |

#### Summary

The AIChatWorkspace CRD accepts `spec.workspaceName` as an arbitrary string with only an immutability CEL rule — no pattern, length, or reserved-name validation. The reconciler uses the field verbatim as the target namespace for every resource it creates (Namespace, ResourceQuota, PVCs, ServiceAccounts, StatefulSet, Deployment, Services, Ingresses) and the finalizer deletion path deletes the namespace bearing that exact name. Combined with the operator's cluster-wide wildcard RBAC, any principal that can create an AIChatWorkspace CR can inject a ResourceQuota (pods: 2, persistentvolumeclaims: 2, services: 5) and full operator-managed workloads into any existing namespace such as `kube-system`, and delete that namespace when the CR is deleted. Multi-label names (dots are legal namespace names) additionally let a tenant claim arbitrary `<x>.<DefaultDomain>` ingress hostnames.

#### Root Cause

The violated invariant is that a namespaced, tenant-controlled CR field must not directly determine the identity of cluster-scoped resources. `spec.workspaceName` is accepted as an arbitrary string (the generated CRD carries only an immutability CEL rule) and is used verbatim as the target namespace for all `ensure*` create operations and as the name of the namespace deleted by the finalizer, while the operator's ClusterRole grants wildcard verbs on `namespaces`, `resourcequotas`, and related core resources across the whole cluster. No ownership, label, or reserved-name check sits between the CR field and these operations.

**workspaceName accepted without validation** — `api/v1alpha1/aichatworkspace_types.go:25-28`

The CRD field is tenant-controlled. The only validation marker is immutability (`self == oldSelf`); there is no pattern, maxLength, or enum, so any string (e.g. `kube-system`, or dotted names up to 253 chars) is accepted by the API server.

```go
	// The name of the workspace.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="WorkspaceName is immutable"
	WorkspaceName string `json:"workspaceName"`
```

**Reconcile targets every resource at spec.WorkspaceName** — `internal/controller/handle_reconcile.go:44-55`

`Spec.WorkspaceName` is used as the Namespace argument for the namespace and the ResourceQuota (and, further down the same function, for the PVC, service accounts, StatefulSet, Deployment, Services, and Ingresses). A CR whose workspaceName equals an existing namespace name therefore writes its resources into that namespace.

```go
	// ensureNamespace - create the "aichatworkspace" namespace that contains all the components required
	// to run the AIChat Workspace.
	namespaceDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.AIChatWorkspaceName)
	result, err = r.ensureNamespace(ctx, aichat, k8s.NewNamespace(aichat.Spec.WorkspaceName, namespaceDefaultLabels))
	if result != nil {
		return result, err
	}

	// ensureResourceQuota - creates the ResourceQuota object that limits resources that can be ran in the namespace.
	resourceQuotaName := generateName(aichat.Spec.WorkspaceName, constants.ResourceQuotaName)
	resourceQuotaDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.ResourceQuotaLabelName)
	result, err = r.ensureResourceQuota(ctx, aichat, k8s.NewResourceQuota(aichat.Spec.WorkspaceName, resourceQuotaName, resourceQuotaDefaultLabels))
```

**ensureNamespace creates if absent, never validates the target** — `internal/controller/namespace.go:41-49`

If the namespace already exists the operator proceeds without any check that it is safe or tenant-owned; subsequent ensure\* calls then create the operator's objects inside it.

```go
	err := r.Get(context.TODO(), types.NamespacedName{
		Name: ns.Name,
	}, found)

	if err != nil && errors.IsNotFound(err) {
		logger.Info("Creating the namespace", "instance.Spec.Namespace", instance.Spec.WorkspaceName)

		controllerutil.SetControllerReference(instance, ns, r.Scheme)
		err = r.Create(context.TODO(), ns)
```

**Delete step removes the finalizer after deleting the namespace** — `internal/controller/delete.go:35-45`

When the CR is deleted and `status.isCreated` is true, the finalizer path calls `deleteAIChatWorkspace` and then removes the finalizer, so the destructive delete is the terminal action of CR deletion.

```go
	if isCreated && pendingDeletion {
		instance.logger.Info("reconciling aichat", "aichat", instance.aichatWorkspaceConfig, "action", "delete")
		if controllerutil.ContainsFinalizer(instance.aichatWorkspaceConfig, aichatWorkspaceFinalizerName) {
			if err = instance.r.deleteAIChatWorkspace(instance.ctx, instance.aichatWorkspaceConfig); err != nil {
				return instance.r.finishReconcile(err, false)
			}
```

**Namespace deleted by spec.WorkspaceName** — `internal/controller/aichatworkspace_controller.go:232-248`

This is the broken control: a cluster-scoped `Delete` on the namespace whose name is the tenant-supplied `Spec.WorkspaceName`, with no ownership, label, or reserved-name check.

```go
func (r *AIChatWorkspaceReconciler) deleteAIChatWorkspace(ctx context.Context, instance *appsv1alpha1.AIChatWorkspace) error {
	logger := log.FromContext(ctx)

	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: instance.Spec.WorkspaceName,
		},
	}
	err := r.Delete(context.TODO(), namespace)
	if err != nil {
		return err
	}

	logger.Info("deleted aichatworkspace", "aichatworkspace", instance.Spec.WorkspaceName, "action", "deleted")

	return nil
}
```

**ResourceQuota hard limits injected into the target namespace** — `internal/adapters/k8s/common.go:169-176`

The quota created in the `Spec.WorkspaceName` namespace caps the whole namespace at 2 pods, 2 PVCs, and 5 services — enough to block new (and rescheduled) workloads in a victim namespace such as kube-system.

```go
		Spec: corev1.ResourceQuotaSpec{
			Hard: corev1.ResourceList{
				"pods":                   resource.MustParse(constants.MaxPods),
				"persistentvolumeclaims": resource.MustParse(constants.MaxPersistentVolumeClaims),
				"services":               resource.MustParse(constants.MaxService),
			},
		},
	}
```

**Quota limit constants** — `internal/constants/constants.go:51-54`

Confirms the injected quota values: a victim namespace is throttled to 2 pods, 2 PVCs, and 5 services.

```go
	ResourceQuotaName         = "rquota"
	MaxPods                   = "2"
	MaxPersistentVolumeClaims = "2"
	MaxService                = "5"
```

**Operator ClusterRole grants wildcard verbs cluster-wide** — `config/rbac/role.yaml:13-24`

The expected control (least privilege) is absent: the manager's ClusterRole has no namespace scoping, so every operation the unvalidated `workspaceName` directs is permitted cluster-wide, including deleting any namespace.

```yaml
- apiGroups:
  - ""
  resources:
  - namespaces
  - persistentvolumeclaims
  - pods
  - pods/exec
  - resourcequotas
  - serviceaccounts
  - services
  verbs:
  - '*'
```

#### Validation

Traced `spec.workspaceName` from the CRD Go type (api/v1alpha1/aichatworkspace_types.go:25-28) and generated CRD schema (only `self == oldSelf`) through `handleReconcile` (handle_reconcile.go:44-131, where every ensure\* call uses `Spec.WorkspaceName` as the object namespace) to `deleteAIChatWorkspace` (aichatworkspace_controller.go:232-248), which issues a cluster-level Delete on the namespace named `Spec.WorkspaceName` once `status.isCreated` is true (delete.go:35-45). The shipped ClusterRole (config/rbac/role.yaml:13-24) confirms the operator SA holds wildcard permissions on namespaces and resourcequotas cluster-wide, and the quota constants (constants.go:51-54) confirm the injected limits.

Validation method: static source trace

**workspaceName accepted without validation** — `api/v1alpha1/aichatworkspace_types.go:25-28`

The CRD field is tenant-controlled. The only validation marker is immutability (`self == oldSelf`); there is no pattern, maxLength, or enum, so any string (e.g. `kube-system`, or dotted names up to 253 chars) is accepted by the API server.

```go
	// The name of the workspace.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="WorkspaceName is immutable"
	WorkspaceName string `json:"workspaceName"`
```

**Reconcile targets every resource at spec.WorkspaceName** — `internal/controller/handle_reconcile.go:44-55`

`Spec.WorkspaceName` is used as the Namespace argument for the namespace and the ResourceQuota (and, further down the same function, for the PVC, service accounts, StatefulSet, Deployment, Services, and Ingresses). A CR whose workspaceName equals an existing namespace name therefore writes its resources into that namespace.

```go
	// ensureNamespace - create the "aichatworkspace" namespace that contains all the components required
	// to run the AIChat Workspace.
	namespaceDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.AIChatWorkspaceName)
	result, err = r.ensureNamespace(ctx, aichat, k8s.NewNamespace(aichat.Spec.WorkspaceName, namespaceDefaultLabels))
	if result != nil {
		return result, err
	}

	// ensureResourceQuota - creates the ResourceQuota object that limits resources that can be ran in the namespace.
	resourceQuotaName := generateName(aichat.Spec.WorkspaceName, constants.ResourceQuotaName)
	resourceQuotaDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.ResourceQuotaLabelName)
	result, err = r.ensureResourceQuota(ctx, aichat, k8s.NewResourceQuota(aichat.Spec.WorkspaceName, resourceQuotaName, resourceQuotaDefaultLabels))
```

**Delete step removes the finalizer after deleting the namespace** — `internal/controller/delete.go:35-45`

When the CR is deleted and `status.isCreated` is true, the finalizer path calls `deleteAIChatWorkspace` and then removes the finalizer, so the destructive delete is the terminal action of CR deletion.

```go
	if isCreated && pendingDeletion {
		instance.logger.Info("reconciling aichat", "aichat", instance.aichatWorkspaceConfig, "action", "delete")
		if controllerutil.ContainsFinalizer(instance.aichatWorkspaceConfig, aichatWorkspaceFinalizerName) {
			if err = instance.r.deleteAIChatWorkspace(instance.ctx, instance.aichatWorkspaceConfig); err != nil {
				return instance.r.finishReconcile(err, false)
			}
```

**Namespace deleted by spec.WorkspaceName** — `internal/controller/aichatworkspace_controller.go:232-248`

This is the broken control: a cluster-scoped `Delete` on the namespace whose name is the tenant-supplied `Spec.WorkspaceName`, with no ownership, label, or reserved-name check.

```go
func (r *AIChatWorkspaceReconciler) deleteAIChatWorkspace(ctx context.Context, instance *appsv1alpha1.AIChatWorkspace) error {
	logger := log.FromContext(ctx)

	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: instance.Spec.WorkspaceName,
		},
	}
	err := r.Delete(context.TODO(), namespace)
	if err != nil {
		return err
	}

	logger.Info("deleted aichatworkspace", "aichatworkspace", instance.Spec.WorkspaceName, "action", "deleted")

	return nil
}
```

**ResourceQuota hard limits injected into the target namespace** — `internal/adapters/k8s/common.go:169-176`

The quota created in the `Spec.WorkspaceName` namespace caps the whole namespace at 2 pods, 2 PVCs, and 5 services — enough to block new (and rescheduled) workloads in a victim namespace such as kube-system.

```go
		Spec: corev1.ResourceQuotaSpec{
			Hard: corev1.ResourceList{
				"pods":                   resource.MustParse(constants.MaxPods),
				"persistentvolumeclaims": resource.MustParse(constants.MaxPersistentVolumeClaims),
				"services":               resource.MustParse(constants.MaxService),
			},
		},
	}
```

**Operator ClusterRole grants wildcard verbs cluster-wide** — `config/rbac/role.yaml:13-24`

The expected control (least privilege) is absent: the manager's ClusterRole has no namespace scoping, so every operation the unvalidated `workspaceName` directs is permitted cluster-wide, including deleting any namespace.

```yaml
- apiGroups:
  - ""
  resources:
  - namespaces
  - persistentvolumeclaims
  - pods
  - pods/exec
  - resourcequotas
  - serviceaccounts
  - services
  verbs:
  - '*'
```

#### Dataflow

spec.workspaceName (CRD, unvalidated) -\> handleReconcile ensure\* calls (namespace = Spec.WorkspaceName) -\> ResourceQuota/PVC/StatefulSet/Deployment/Service/Ingress created in the chosen namespace -\> deleteAIChatWorkspace Delete(namespace=Spec.WorkspaceName) on CR deletion

- **Source:** attacker-controlled spec.workspaceName string

- **Sink:** cluster-scoped namespace Delete and ResourceQuota/workload Create in any namespace

- **Outcome:** arbitrary namespace deletion (e.g. kube-system) and ResourceQuota-based availability loss in any namespace

**Reconcile targets every resource at spec.WorkspaceName** — `internal/controller/handle_reconcile.go:44-55`

`Spec.WorkspaceName` is used as the Namespace argument for the namespace and the ResourceQuota (and, further down the same function, for the PVC, service accounts, StatefulSet, Deployment, Services, and Ingresses). A CR whose workspaceName equals an existing namespace name therefore writes its resources into that namespace.

```go
	// ensureNamespace - create the "aichatworkspace" namespace that contains all the components required
	// to run the AIChat Workspace.
	namespaceDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.AIChatWorkspaceName)
	result, err = r.ensureNamespace(ctx, aichat, k8s.NewNamespace(aichat.Spec.WorkspaceName, namespaceDefaultLabels))
	if result != nil {
		return result, err
	}

	// ensureResourceQuota - creates the ResourceQuota object that limits resources that can be ran in the namespace.
	resourceQuotaName := generateName(aichat.Spec.WorkspaceName, constants.ResourceQuotaName)
	resourceQuotaDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.ResourceQuotaLabelName)
	result, err = r.ensureResourceQuota(ctx, aichat, k8s.NewResourceQuota(aichat.Spec.WorkspaceName, resourceQuotaName, resourceQuotaDefaultLabels))
```

**Namespace deleted by spec.WorkspaceName** — `internal/controller/aichatworkspace_controller.go:232-248`

This is the broken control: a cluster-scoped `Delete` on the namespace whose name is the tenant-supplied `Spec.WorkspaceName`, with no ownership, label, or reserved-name check.

```go
func (r *AIChatWorkspaceReconciler) deleteAIChatWorkspace(ctx context.Context, instance *appsv1alpha1.AIChatWorkspace) error {
	logger := log.FromContext(ctx)

	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: instance.Spec.WorkspaceName,
		},
	}
	err := r.Delete(context.TODO(), namespace)
	if err != nil {
		return err
	}

	logger.Info("deleted aichatworkspace", "aichatworkspace", instance.Spec.WorkspaceName, "action", "deleted")

	return nil
}
```

**Operator ClusterRole grants wildcard verbs cluster-wide** — `config/rbac/role.yaml:13-24`

The expected control (least privilege) is absent: the manager's ClusterRole has no namespace scoping, so every operation the unvalidated `workspaceName` directs is permitted cluster-wide, including deleting any namespace.

```yaml
- apiGroups:
  - ""
  resources:
  - namespaces
  - persistentvolumeclaims
  - pods
  - pods/exec
  - resourcequotas
  - serviceaccounts
  - services
  verbs:
  - '*'
```

**ResourceQuota hard limits injected into the target namespace** — `internal/adapters/k8s/common.go:169-176`

The quota created in the `Spec.WorkspaceName` namespace caps the whole namespace at 2 pods, 2 PVCs, and 5 services — enough to block new (and rescheduled) workloads in a victim namespace such as kube-system.

```go
		Spec: corev1.ResourceQuotaSpec{
			Hard: corev1.ResourceList{
				"pods":                   resource.MustParse(constants.MaxPods),
				"persistentvolumeclaims": resource.MustParse(constants.MaxPersistentVolumeClaims),
				"services":               resource.MustParse(constants.MaxService),
			},
		},
	}
```

#### Reachability

Requires only the standard tenant grant of create on aichatworkspaces in any namespace, plus the operator deployed with the shipped ClusterRole. No special name format is needed: `kube-system` is a legal namespace name.

- **Attacker:** any principal with write access to the AIChatWorkspace CRD (a normal multi-tenant grant)

- **Entry point:** create/update of an AIChatWorkspace custom resource

- **Outcome:** a cluster namespace is destroyed, or its workload admission is throttled to 2 pods / 2 PVCs / 5 services, or operator-managed containers run inside the victim namespace

#### Severity

**High** — A single CR create (or delete) by any user with write access to the namespaced CRD can delete an arbitrary cluster namespace — including kube-system, destroying all system workloads — or DoS any namespace by adding a pods:2 ResourceQuota, and can run operator-managed ollama/Open WebUI containers inside the victim namespace. Impact is cluster-wide destruction or availability loss. Likelihood is high: no crafted payload is needed, `kube-system` is a legal namespace name, and CR write access is a normal tenant grant.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Validate `spec.workspaceName` against an RFC 1123 label (`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`, maxLength 63), reject reserved/system namespace names (kube-system, kube-public, kube-node-lease, and the operator's own system namespace), and gate `deleteAIChatWorkspace` so it only deletes namespaces the operator created (identified by an ownership label or controller reference).

Tests:
- Create a CR with spec.workspaceName=kube-system and assert the operator neither creates a ResourceQuota/workloads in kube-system nor deletes the namespace on CR deletion.
- Assert the CRD rejects workspaceName values containing '.', values over 63 characters, and reserved system namespace names.
- Assert the finalizer deletion path only deletes namespaces carrying the operator's ownership label or controller reference.

Preventive controls:
- Add CRD validation (pattern, maxLength) plus a reserved-namespace deny list for spec.workspaceName.
- Tag operator-created namespaces with an ownership label and restrict deleteAIChatWorkspace to namespaces it created.
- Scope the operator ClusterRole where possible (e.g. exclude system namespaces) so wildcard verbs do not apply to cluster-critical resources.

<a id="finding-2"></a>

### [2] Cross-tenant workspace-name collision exposes another tenant's unauthenticated Open WebUI

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | medium |
| Confidence rationale | The source establishes the namespaced-CR to cluster-global-namespace mapping (CRD scope Namespaced; ensureNamespace keyed on Spec.WorkspaceName), the predictable hostname derivation (setIngressDNSHost), and the absence of any authentication configuration on the Open WebUI deployment; real-world likelihood depends on the deployment's naming practices and DNS exposure. |
| Category | Access Control |
| CWE | CWE-639, CWE-306 |
| Affected lines | api/v1alpha1/aichatworkspace_types.go:25-28, config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:15, internal/controller/handle_reconcile.go:46-47, internal/controller/handle_reconcile.go:153-163, internal/adapters/k8s/apps.go:76-97 |

#### Summary

AIChatWorkspace CRs are namespaced, but `spec.workspaceName` maps to a cluster-scoped namespace and a public ingress hostname with no uniqueness enforcement. Two tenants that choose the same workspaceName target the same namespace: the first reconcile wins the workload names, and the Open WebUI it serves — deployed with no authentication at `<name>.<DefaultDomain>` — becomes reachable by the second tenant (and anyone who knows the name), disclosing the first tenant's chat history, uploaded documents, and Ollama model configuration.

#### Root Cause

The violated invariant is that each tenant's workspace must be addressable only by that tenant. A namespaced CR's cluster-global namespace and ingress hostname are keyed solely on the tenant-chosen `spec.workspaceName` with no uniqueness check, and the workload serving that hostname (Open WebUI) is deployed without authentication, so anyone who collides on or learns the name reaches another tenant's data.

**CR is namespaced, workspace identity is cluster-global** — `config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:15`

The CRD is namespaced, so two tenants in different namespaces can both create CRs; nothing scopes the tenant boundary to the CR's namespace for the resources derived from `spec.workspaceName`.

```yaml
  scope: Namespaced
```

**Target namespace is the tenant-chosen name** — `internal/controller/handle_reconcile.go:46-47`

The namespace — a cluster-scoped resource — is named solely by `Spec.WorkspaceName`. Two CRs from different tenant namespaces with the same value collide on the same namespace, and the first to reconcile owns the workload names inside it.

```go
	namespaceDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.AIChatWorkspaceName)
	result, err = r.ensureNamespace(ctx, aichat, k8s.NewNamespace(aichat.Spec.WorkspaceName, namespaceDefaultLabels))
```

**Public hostname derived from the same field** — `internal/controller/handle_reconcile.go:153-163`

The ingress host `<name>.<DefaultDomain>` is a deterministic function of the tenant-chosen workspace name, so a colliding (or name-guessing) tenant knows the exact URL of the victim's webui.

```go
func setIngressDNSHost(config *config.Config, workspace string, workload string) string {
	var dnsName string
	switch workload {
	case "ollama":
		dnsName = fmt.Sprintf("%s-api.%s", workspace, config.DefaultDomain)
	case "openwebui":
		dnsName = fmt.Sprintf("%s.%s", workspace, config.DefaultDomain)
	}

	return dnsName
}
```

**Open WebUI deployment carries no authentication** — `internal/adapters/k8s/apps.go:76-97`

The broken control is the missing authentication: the webui container env configures no admin user, SSO, or auth proxy, so anyone who reaches the ingress host is admitted to the workspace's data.

```go
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_BASE_URL",
									Value: ollamaServerURI,
								},
								{
									Name:  "OPENAI_API_BASE_URL",
									Value: openAIURI,
								},
								{
									Name:  "ENV",
									Value: "dev",
								},
								{
									Name:  "WEBUI_NAME",
									Value: workspaceName,
								},
								{
									Name:  "KEY_FILE",
									Value: "/tmp/.webui_secret_key",
								},
							},
```

#### Validation

The CRD is namespaced (config/crd/bases/...yaml:15) while ensureNamespace/ensureIngress key cluster-scoped and public resources solely on Spec.WorkspaceName (handle_reconcile.go:46-47); setIngressDNSHost (handle_reconcile.go:153-163) derives the public hostname `<name>.<DefaultDomain>` from the same field; NewDeployment (apps.go:76-97) configures Open WebUI with no authentication settings, and no auth layer sits between the ingress and the webui service.

Validation method: static source trace

**CR is namespaced, workspace identity is cluster-global** — `config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:15`

The CRD is namespaced, so two tenants in different namespaces can both create CRs; nothing scopes the tenant boundary to the CR's namespace for the resources derived from `spec.workspaceName`.

```yaml
  scope: Namespaced
```

**Target namespace is the tenant-chosen name** — `internal/controller/handle_reconcile.go:46-47`

The namespace — a cluster-scoped resource — is named solely by `Spec.WorkspaceName`. Two CRs from different tenant namespaces with the same value collide on the same namespace, and the first to reconcile owns the workload names inside it.

```go
	namespaceDefaultLabels := defaultLabels(aichat.Spec.WorkspaceName, aichat.Spec.WorkspaceName, constants.AIChatWorkspaceName)
	result, err = r.ensureNamespace(ctx, aichat, k8s.NewNamespace(aichat.Spec.WorkspaceName, namespaceDefaultLabels))
```

**Public hostname derived from the same field** — `internal/controller/handle_reconcile.go:153-163`

The ingress host `<name>.<DefaultDomain>` is a deterministic function of the tenant-chosen workspace name, so a colliding (or name-guessing) tenant knows the exact URL of the victim's webui.

```go
func setIngressDNSHost(config *config.Config, workspace string, workload string) string {
	var dnsName string
	switch workload {
	case "ollama":
		dnsName = fmt.Sprintf("%s-api.%s", workspace, config.DefaultDomain)
	case "openwebui":
		dnsName = fmt.Sprintf("%s.%s", workspace, config.DefaultDomain)
	}

	return dnsName
}
```

**Open WebUI deployment carries no authentication** — `internal/adapters/k8s/apps.go:76-97`

The broken control is the missing authentication: the webui container env configures no admin user, SSO, or auth proxy, so anyone who reaches the ingress host is admitted to the workspace's data.

```go
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_BASE_URL",
									Value: ollamaServerURI,
								},
								{
									Name:  "OPENAI_API_BASE_URL",
									Value: openAIURI,
								},
								{
									Name:  "ENV",
									Value: "dev",
								},
								{
									Name:  "WEBUI_NAME",
									Value: workspaceName,
								},
								{
									Name:  "KEY_FILE",
									Value: "/tmp/.webui_secret_key",
								},
							},
```

#### Dataflow

spec.workspaceName (both CRs) -\> shared namespace and \<name\>.\<DefaultDomain\> ingress host -\> unauthenticated Open WebUI of the first tenant

- **Source:** second tenant's identical (or guessed) spec.workspaceName

- **Sink:** first tenant's Open WebUI HTTP service

- **Outcome:** cross-tenant disclosure of workspace data and uploads

**CR is namespaced, workspace identity is cluster-global** — `config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:15`

The CRD is namespaced, so two tenants in different namespaces can both create CRs; nothing scopes the tenant boundary to the CR's namespace for the resources derived from `spec.workspaceName`.

```yaml
  scope: Namespaced
```

**Public hostname derived from the same field** — `internal/controller/handle_reconcile.go:153-163`

The ingress host `<name>.<DefaultDomain>` is a deterministic function of the tenant-chosen workspace name, so a colliding (or name-guessing) tenant knows the exact URL of the victim's webui.

```go
func setIngressDNSHost(config *config.Config, workspace string, workload string) string {
	var dnsName string
	switch workload {
	case "ollama":
		dnsName = fmt.Sprintf("%s-api.%s", workspace, config.DefaultDomain)
	case "openwebui":
		dnsName = fmt.Sprintf("%s.%s", workspace, config.DefaultDomain)
	}

	return dnsName
}
```

**Open WebUI deployment carries no authentication** — `internal/adapters/k8s/apps.go:76-97`

The broken control is the missing authentication: the webui container env configures no admin user, SSO, or auth proxy, so anyone who reaches the ingress host is admitted to the workspace's data.

```go
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_BASE_URL",
									Value: ollamaServerURI,
								},
								{
									Name:  "OPENAI_API_BASE_URL",
									Value: openAIURI,
								},
								{
									Name:  "ENV",
									Value: "dev",
								},
								{
									Name:  "WEBUI_NAME",
									Value: workspaceName,
								},
								{
									Name:  "KEY_FILE",
									Value: "/tmp/.webui_secret_key",
								},
							},
```

#### Reachability

Requires a second CR with a colliding name plus network reachability to the ingress host; no cluster privileges beyond CR write are needed.

- **Attacker:** second tenant (or an external visitor who learns or guesses the workspace name)

- **Entry point:** creating an AIChatWorkspace CR with a colliding spec.workspaceName, or visiting the victim's hostname

- **Outcome:** full read (and write) access to the other tenant's workspace data

#### Severity

**Medium** — Cross-tenant disclosure of a personal AI workspace's data and uploads. Likelihood is medium: it requires two tenants to pick the same (or guessable) workspaceName — likely with short, human-predictable names such as usernames — and the webui has no authentication, so knowledge of the name suffices. Impact is medium: disclosure of one workspace's data without cluster-level effect.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Enforce cluster-wide uniqueness of `spec.workspaceName` (e.g. a validating admission webhook or a cluster-scoped name lock object) and/or derive the target namespace from both the CR namespace and the name. Additionally require authentication on the exposed Open WebUI before public ingress traffic reaches it.

Tests:
- Assert that two CRs in different namespaces with the same spec.workspaceName are rejected or mapped to distinct namespaces and hostnames.
- Assert the Open WebUI deployment enforces authentication before the ingress routes traffic.

Preventive controls:
- Add a validating webhook (or cluster-scoped name lock) enforcing global uniqueness of spec.workspaceName.
- Derive the target namespace from both the CR namespace and workspaceName.
- Enable Open WebUI authentication (admin user or SSO) as a default.

<a id="finding-3"></a>

### [3] Shipped manifests embed weak, hardcoded database credentials

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | The literal values are present in the committed manifests and wired into the deployment via env vars and secretKeyRef, and the default kustomize overlay includes both files. |
| Category | Sensitive Data Exposure |
| CWE | CWE-798, CWE-1393 |
| Affected lines | config/default/mysql-deployment.yaml:22-30, config/default/mysql-secret.yaml:5-7, config/default/kustomization.yaml:35-39 |

#### Summary

The default deployment overlay ships a MySQL deployment whose application user password is the literal string `aichatworkspace` in plaintext, and a Secret whose root password is the literal string `password`. Both files are included by `config/default/kustomization.yaml`, so a stock install provisions a database with publicly known, dictionary-grade credentials.

#### Root Cause

The violated invariant is that shipped deployment manifests must not embed working credentials. `config/default/mysql-deployment.yaml` hardcodes the application user password (`MYSQL_PASSWORD: aichatworkspace`) and points the root password at `config/default/mysql-secret.yaml`, which stores the literal `password`; `config/default/kustomization.yaml` includes both files, so a stock install provisions a database whose credentials are publicly known and trivially guessable.

**Application DB user with plaintext password** — `config/default/mysql-deployment.yaml:22-30`

The broken control is the credential itself: `MYSQL_PASSWORD` is the literal string `aichatworkspace` (identical to the username), committed in plaintext in the deployment manifest.

```yaml
        - name: MYSQL_USER
          value: aichatworkspace
        - name: MYSQL_PASSWORD
          value: "aichatworkspace"
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password
```

**Root password stored as the literal 'password'** — `config/default/mysql-secret.yaml:5-7`

The Secret referenced by `MYSQL_ROOT_PASSWORD` stores the literal password `password`, so the database root account is also publicly known.

```yaml
type: kubernetes.io/basic-auth
stringData:
  password: password
```

**Default overlay installs both files** — `config/default/kustomization.yaml:35-39`

The `config/default` overlay — the stock install surface — includes both the Secret and the deployment, so the weak credentials apply to a default apply.

```yaml
- system-configmap.yaml
- mysql-secret.yaml
- mysql-storage.yaml
- mysql-deployment.yaml
- mysql-service.yaml
```

#### Validation

Confirmed the literal values in the committed YAML: MYSQL_PASSWORD is the string aichatworkspace (mysql-deployment.yaml:24-25), the Secret's stringData.password is the string password (mysql-secret.yaml:7), and the default kustomize overlay lists both files (kustomization.yaml:36,38).

Validation method: static manifest inspection

**Application DB user with plaintext password** — `config/default/mysql-deployment.yaml:22-30`

The broken control is the credential itself: `MYSQL_PASSWORD` is the literal string `aichatworkspace` (identical to the username), committed in plaintext in the deployment manifest.

```yaml
        - name: MYSQL_USER
          value: aichatworkspace
        - name: MYSQL_PASSWORD
          value: "aichatworkspace"
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password
```

**Root password stored as the literal 'password'** — `config/default/mysql-secret.yaml:5-7`

The Secret referenced by `MYSQL_ROOT_PASSWORD` stores the literal password `password`, so the database root account is also publicly known.

```yaml
type: kubernetes.io/basic-auth
stringData:
  password: password
```

**Default overlay installs both files** — `config/default/kustomization.yaml:35-39`

The `config/default` overlay — the stock install surface — includes both the Secret and the deployment, so the weak credentials apply to a default apply.

```yaml
- system-configmap.yaml
- mysql-secret.yaml
- mysql-storage.yaml
- mysql-deployment.yaml
- mysql-service.yaml
```

#### Dataflow

committed manifest literals -\> installed Secret and deployment env -\> MySQL accounts with known credentials -\> database access

- **Source:** repository-committed credential literals

- **Sink:** MySQL authentication (user aichatworkspace or root)

- **Outcome:** unauthorized read/write of the workspace database

**Application DB user with plaintext password** — `config/default/mysql-deployment.yaml:22-30`

The broken control is the credential itself: `MYSQL_PASSWORD` is the literal string `aichatworkspace` (identical to the username), committed in plaintext in the deployment manifest.

```yaml
        - name: MYSQL_USER
          value: aichatworkspace
        - name: MYSQL_PASSWORD
          value: "aichatworkspace"
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password
```

**Root password stored as the literal 'password'** — `config/default/mysql-secret.yaml:5-7`

The Secret referenced by `MYSQL_ROOT_PASSWORD` stores the literal password `password`, so the database root account is also publicly known.

```yaml
type: kubernetes.io/basic-auth
stringData:
  password: password
```

**Default overlay installs both files** — `config/default/kustomization.yaml:35-39`

The `config/default` overlay — the stock install surface — includes both the Secret and the deployment, so the weak credentials apply to a default apply.

```yaml
- system-configmap.yaml
- mysql-secret.yaml
- mysql-storage.yaml
- mysql-deployment.yaml
- mysql-service.yaml
```

#### Reachability

Requires network or cluster access to the MySQL service, or repository/manifest access to learn the defaults; no further privileges are needed.

- **Attacker:** any principal with network or cluster access to the MySQL service (or repository access to the manifests)

- **Entry point:** MySQL authentication with the shipped account names and passwords

- **Outcome:** database access with the shipped accounts' privileges

#### Severity

**Medium** — Hardcoded, weak credentials for a database that can store workspace data. Impact is medium (known credentials grant database access to anyone who can reach the service or read the repository/manifests); likelihood is medium because the values are public in the repository and the default overlay applies them to real installs.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Generate strong random credentials at install time (e.g. kustomize `secretGenerator` or an external secret store), remove the literal `aichatworkspace` and `password` values from the repository, and rotate credentials in any deployment that used the shipped defaults.

Tests:
- Assert the rendered default manifests contain no literal password values.
- Assert the mysql-secret value is generated randomly at install time.

Preventive controls:
- Use kustomize secretGenerator (or an external secret store) for all database credentials.
- Document credential rotation for clusters installed from the shipped defaults.

<a id="finding-4"></a>

### [4] Open WebUI workload is deployed as root without a security context

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The commented-out security-context lines and the absence of any runAsNonRoot/capability/read-only-filesystem settings on the webui container are directly visible in source, while the Ollama StatefulSet in the same file applies the hardening helpers. |
| Category | Security Misconfiguration |
| CWE | CWE-250, CWE-16 |
| Affected lines | internal/adapters/k8s/apps.go:65-75, internal/adapters/k8s/apps.go:184-199 |

#### Summary

`NewDeployment` for Open WebUI ships with both the pod and container SecurityContexts commented out, so the network-exposed, unauthenticated webui container runs as root with default capabilities and a writable root filesystem. The sibling Ollama StatefulSet in the same file applies the hardened `defaultPodSecurityContext()`/`defaultSecurityContext()` helpers, showing the intended baseline that the webui path omits.

#### Root Cause

The violated invariant is that network-reachable operator-managed workloads should run under the operator's own hardened baseline. `NewDeployment` comments out the pod and container security contexts and never sets RunAsNonRoot, capability drops, or a read-only root filesystem, so the webui container keeps root privileges and default capabilities — while the Ollama StatefulSet in the same file applies `defaultPodSecurityContext()` and `defaultSecurityContext()`.

**Webui pod/container security contexts commented out** — `internal/adapters/k8s/apps.go:65-75`

The broken control: both the pod and container security contexts are commented out (with a note about non-root image issues), so the webui container keeps root privileges, default capabilities, and a writable root filesystem.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					// at the moment having issues getting open webui to run as non-root
					// SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OpenwebuiContainerName,
							Image: containerImage,
							// SecurityContext: defaultSecurityContext(),
```

**Ollama StatefulSet applies the hardened baseline** — `internal/adapters/k8s/apps.go:184-199`

The expected control exists in the same file: the Ollama pod spec applies defaultPodSecurityContext() (uid/gid 10001) and the container applies defaultSecurityContext() (non-root, no privilege escalation, all capabilities dropped, read-only root filesystem), which the webui path omits.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OllamaContainerName,
							Image: containerImage,
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_DEBUG",
									Value: "1",
								},
							},
							SecurityContext: defaultSecurityContext(),
```

#### Validation

Confirmed the webui PodSpec and container carry no security context (apps.go:65-75, both lines commented out with a TODO-style note) while the Ollama StatefulSet (apps.go:184-199) applies the hardened helpers defined at apps.go:237-267.

Validation method: static source inspection

**Webui pod/container security contexts commented out** — `internal/adapters/k8s/apps.go:65-75`

The broken control: both the pod and container security contexts are commented out (with a note about non-root image issues), so the webui container keeps root privileges, default capabilities, and a writable root filesystem.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					// at the moment having issues getting open webui to run as non-root
					// SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OpenwebuiContainerName,
							Image: containerImage,
							// SecurityContext: defaultSecurityContext(),
```

**Ollama StatefulSet applies the hardened baseline** — `internal/adapters/k8s/apps.go:184-199`

The expected control exists in the same file: the Ollama pod spec applies defaultPodSecurityContext() (uid/gid 10001) and the container applies defaultSecurityContext() (non-root, no privilege escalation, all capabilities dropped, read-only root filesystem), which the webui path omits.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OllamaContainerName,
							Image: containerImage,
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_DEBUG",
									Value: "1",
								},
							},
							SecurityContext: defaultSecurityContext(),
```

#### Dataflow

webui HTTP request -\> application vulnerability -\> root container process with default capabilities

- **Source:** network attacker reaching the unauthenticated webui

- **Sink:** root-equivalent code execution inside the webui container

- **Outcome:** elevated foothold for node escape or persistence

**Webui pod/container security contexts commented out** — `internal/adapters/k8s/apps.go:65-75`

The broken control: both the pod and container security contexts are commented out (with a note about non-root image issues), so the webui container keeps root privileges, default capabilities, and a writable root filesystem.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					// at the moment having issues getting open webui to run as non-root
					// SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OpenwebuiContainerName,
							Image: containerImage,
							// SecurityContext: defaultSecurityContext(),
```

**Ollama StatefulSet applies the hardened baseline** — `internal/adapters/k8s/apps.go:184-199`

The expected control exists in the same file: the Ollama pod spec applies defaultPodSecurityContext() (uid/gid 10001) and the container applies defaultSecurityContext() (non-root, no privilege escalation, all capabilities dropped, read-only root filesystem), which the webui path omits.

```go
				Spec: v1.PodSpec{
					RestartPolicy:                v1.RestartPolicyAlways,
					ServiceAccountName:           saName,
					AutomountServiceAccountToken: ptr.To[bool](false),
					SecurityContext:              defaultPodSecurityContext(),
					Containers: []v1.Container{
						{
							Name:  constants.OllamaContainerName,
							Image: containerImage,
							Env: []v1.EnvVar{
								{
									Name:  "OLLAMA_DEBUG",
									Value: "1",
								},
							},
							SecurityContext: defaultSecurityContext(),
```

#### Reachability

Requires a code-execution primitive in the Open WebUI application; the missing hardening then determines how far that primitive reaches.

- **Attacker:** attacker who compromises the Open WebUI application

- **Entry point:** Open WebUI HTTP service

- **Outcome:** root-equivalent code execution inside the webui container

#### Severity

**Low** — Weakens the container-escape boundary for a network-reachable third-party application; exploitation still requires a webui RCE, and the blast radius is the container/node boundary rather than the cluster.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Enable the existing `defaultPodSecurityContext()` and `defaultSecurityContext()` (or equivalent restricted Pod Security Standard settings) on the Open WebUI deployment, resolving the non-root image concerns noted in the code comments.

Tests:
- Assert the rendered Open WebUI deployment has runAsNonRoot, dropped capabilities, and a read-only root filesystem.

Preventive controls:
- Apply restricted Pod Security Standards to workspace namespaces.
- Adopt the same security-context helpers for all operator-managed workloads.

<a id="finding-5"></a>

### [5] Manager exposes unauthenticated pprof/expvar/statsviz debug endpoints and development logging

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The unauthenticated mux and `localhost:6060` listener, plus `Development: true`, are directly present in the source with no conditional gating. |
| Category | Sensitive Data Exposure |
| CWE | CWE-200 |
| Affected lines | cmd/main.go:112-123, cmd/main.go:83-85 |

#### Summary

`cmd/main.go` unconditionally starts an additional HTTP server on `localhost:6060` exposing `net/http/pprof` (including `/debug/pprof/cmdline`), `expvar`, and `statsviz` with no authentication, and initializes the zap logger with `Development: true`. Any code sharing the manager pod's network namespace (a sidecar, a host-network peer, or a `pods/exec` session) can read runtime profiles, command-line arguments, and internal statistics from this unauthenticated surface.

#### Root Cause

The violated invariant is that a production manager must not serve unauthenticated debug interfaces. `main()` unconditionally starts an HTTP server on `localhost:6060` wiring pprof (including `/debug/pprof/cmdline`), expvar, and statsviz with no authn/authz, and sets the zap logger to `Development: true`, which raises log verbosity.

**Unauthenticated pprof/expvar/statsviz server** — `cmd/main.go:112-123`

The broken control: the mux is wired with no authentication or authorization and served unconditionally on port 6060; `/debug/pprof/cmdline` additionally discloses the process's command-line arguments.

```go
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		mux.Handle("/debug/vars/", expvar.Handler())

		statsviz.Register(mux)
		http.ListenAndServe("localhost:6060", mux)
	}()
```

**Development-mode logging enabled** — `cmd/main.go:83-85`

`Development: true` raises log verbosity in the production manager, increasing the chance that request or state details land in logs that co-located code can read.

```go
	opts := zap.Options{
		Development: true,
	}
```

#### Validation

Confirmed the debug server starts unconditionally in main() (cmd/main.go:112-123) with pprof/expvar/statsviz handlers and no middleware, and the zap options set Development: true (cmd/main.go:83-85).

Validation method: static source inspection

**Unauthenticated pprof/expvar/statsviz server** — `cmd/main.go:112-123`

The broken control: the mux is wired with no authentication or authorization and served unconditionally on port 6060; `/debug/pprof/cmdline` additionally discloses the process's command-line arguments.

```go
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		mux.Handle("/debug/vars/", expvar.Handler())

		statsviz.Register(mux)
		http.ListenAndServe("localhost:6060", mux)
	}()
```

**Development-mode logging enabled** — `cmd/main.go:83-85`

`Development: true` raises log verbosity in the production manager, increasing the chance that request or state details land in logs that co-located code can read.

```go
	opts := zap.Options{
		Development: true,
	}
```

#### Dataflow

same-pod network access -\> localhost:6060 pprof/expvar/statsviz -\> runtime state, profiles, and command line

- **Source:** attacker code inside the manager pod's network namespace

- **Sink:** unauthenticated debug HTTP endpoints

- **Outcome:** disclosure of the manager's runtime state and arguments

**Unauthenticated pprof/expvar/statsviz server** — `cmd/main.go:112-123`

The broken control: the mux is wired with no authentication or authorization and served unconditionally on port 6060; `/debug/pprof/cmdline` additionally discloses the process's command-line arguments.

```go
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		mux.Handle("/debug/vars/", expvar.Handler())

		statsviz.Register(mux)
		http.ListenAndServe("localhost:6060", mux)
	}()
```

**Development-mode logging enabled** — `cmd/main.go:83-85`

`Development: true` raises log verbosity in the production manager, increasing the chance that request or state details land in logs that co-located code can read.

```go
	opts := zap.Options{
		Development: true,
	}
```

#### Reachability

Requires an existing foothold in the manager pod (sidecar or exec); the debug server then amplifies it with unauthenticated introspection.

- **Attacker:** code executing in the manager pod's network namespace

- **Entry point:** HTTP request to localhost:6060

- **Outcome:** unauthenticated access to runtime profiles, expvar metrics, and the process command line

#### Severity

**Low** — Information disclosure and runtime profiling confined to the manager pod's network namespace; the main container is distroless nonroot and no sidecar is shipped, so reaching the endpoint requires an additional foothold such as pod exec or a co-located container.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Gate the debug server behind a build tag or an explicit flag and disable it by default in production images; use a production zap configuration instead of `Development: true`.

Tests:
- Assert the manager binary exposes no /debug/pprof endpoint when started without an explicit debug flag.
- Assert the default log configuration is production-grade (Development: false).

Preventive controls:
- Build the debug server into a debug-only binary or guard it with a flag.
- Use a production zap configuration for the shipped image.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| CRD API types, kubebuilder validation markers, and CRD YAML (api/v1alpha1, config/crd) | not recorded | Reported | Reviewed all CRD spec fields and validation markers. spec.workspaceName is only marked Required+Immutable (CEL self == oldSelf) with no format/pattern/length constraints and is used verbatim as the target namespace; see findings input-validation.workspace-namespace-control and access-control.workspace-name-collision. |
| Controller reconcile, create/update/delete, and finalizer logic (internal/controller) | not recorded | Reported | Reviewed reconcile flow, deleteAIChatWorkspace, and finalizer add/remove. The CR deletion path deletes the namespace named by the unvalidated spec.workspaceName, enabling arbitrary namespace deletion with the operator's wildcard RBAC; see finding input-validation.workspace-namespace-control. |
| k8s adapter that creates workspace resources (internal/adapters/k8s) | not recorded | Reported | Reviewed common.go ensure\* helpers and apps.go Deployment/Service builders. webui Deployment has its SecurityContext blocks commented out (PodSecurityContext, runAsNonRoot, allowPrivilegeEscalation false, seccompProfile) while the ollama Deployment is hardened; see finding misconfiguration.webui-security-context. Ingress host is derived from the unvalidated workspace name (see finding access-control.workspace-name-collision). |
| Ingress, Service, and external exposure of workspace web UI | not recorded | Reported | Reviewed ingress.go, service.go, and open-webui.go. Ingress host \<workspaceName\>.\<DefaultDomain\> is derived from an unvalidated, tenant-controlled field; webui is served unauthenticated, so any second tenant choosing the same name reaches the first tenant's web UI; see finding access-control.workspace-name-collision. Note: ensureIngress does not set a controller owner reference (tracked as open question). |
| Operator RBAC, service account, and cluster privileges (config/rbac) | not recorded | Reported | Reviewed role.yaml and role_binding.yaml. The operator ClusterRole uses wildcard resources/verbs cluster-wide, including namespaces and events, which is what elevates the unvalidated workspaceName input into cluster-level impact; see finding input-validation.workspace-namespace-control. |
| Credentials, secrets, and credential defaults (config/default, internal/config) | not recorded | Reported | Reviewed mysql-secret.yaml, mysql-deployment.yaml, and the default kustomization overlay. Default credentials (MYSQL_PASSWORD="aichatworkspace", secret key password: "password") are shipped in manifests included by the default overlay; see finding hardcoded-credentials.mysql-defaults. |
| Container images, Dockerfile, manager deployment, and runtime configuration | not recorded | Reported | Reviewed Dockerfile (distroless static, nonroot user) and config/manager/manager.yaml (runAsNonRoot, seccompRuntimeDefault, capabilities dropped). Image and manager hardening is adequate; the webui workload security-context gap is reported separately. |
| Manager entrypoint and embedded HTTP/debug servers (cmd/main.go) | not recorded | Reported | cmd/main.go starts a debug HTTP server (pprof, expvar, statsviz) on localhost:6060 and passes Development: true to the manager; reachable from anything that obtains a pod foothold; see finding sensitive-data-exposure.debug-endpoints. |
| Ollama adapter (internal/adapters/ollama) | not recorded | No issue found | Reviewed ollama.go. Base URL is always the workspace's own in-cluster Ollama service derived from the operator's own reconciled resources, so no externally controllable URL reaches the adapter (no SSRF path). Model pulls are tenant-directed but scoped to the tenant's own instance and storage (tracked as open question). |
| Modelfile template generation and prompt content (internal/adapters/ai) | not recorded | No issue found | Reviewed modelfiles.go. spec.patterns strings are interpolated verbatim into ollama Modelfiles via fmt.Sprintf, allowing a tenant to escape the SYSTEM block, but the resulting model only affects models in the tenant's own workspace (tracked as open question). No cross-tenant or cluster-level effect identified. |
| Dockerfile, Makefile, build targets, and GitHub workflows | not recorded | No issue found | Reviewed Dockerfile, Makefile, and .github/workflows. No piped remote script execution, no credential exposure in CI, and no obvious supply-chain tampering path; workflow permissions use needs.\* outputs but are not a privilege escalation vector in this repo's context. |
| Unit and e2e test fixtures (test/, \*_test.go) | not recorded | No issue found | Reviewed test/ fixtures (chainsaw workspace.yaml) and unit tests. Test data contains no credentials or secret material beyond the already-reported default MySQL fixtures in config/default. |
| Documentation, hack manifests, and project metadata | not recorded | No issue found | Reviewed docs, hack/, and project metadata. No embedded credentials or risky instructions; README usage examples match the shipped manifests. |

## Open Questions And Follow Up

- CRs that never reach status.isCreated=true retain their finalizer forever and appear stuck in Terminating (the finalizer is only removed in the isCreated && pendingDeletion path).
- ExecuteRemoteCommand loads kubeconfig via clientcmd default loading rules instead of the in-cluster config; inside the manager pod this likely fails, which would block status.isCreated from ever being set. Verify the intended deployment configuration.
- spec.patterns strings are interpolated verbatim into ollama Modelfiles via fmt.Sprintf, letting a tenant escape the SYSTEM block and control Modelfile directives — but only for models in their own workspace. Confirm this is acceptable.
- spec.models entries trigger ollama pulls in the workspace's own Ollama instance; a tenant can pull arbitrarily large models and exhaust the 20Gi PVC or cluster egress. Consider a size/count limit.
- ensureIngress does not set a controller owner reference on Ingress objects (unlike every other ensure\* path), orphaning them if the namespace survives CR removal (e.g. operator uninstalled).
