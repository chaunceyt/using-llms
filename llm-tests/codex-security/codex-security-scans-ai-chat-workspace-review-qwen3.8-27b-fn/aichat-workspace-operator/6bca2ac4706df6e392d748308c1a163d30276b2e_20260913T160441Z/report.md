# Security Review: aichat-workspace-operator

## Scope

The scan reviewed the canonical include paths and exclusions listed below.

- Scan mode: repository
- Target kind: git_revision
- Target ID: github.com/chaunceyt/aichat-workspace-operator@6bca2ac4706df6e392d748308c1a163d30276b2e
- Revision: 6bca2ac4706df6e392d748308c1a163d30276b2e
- Inventory strategy: repository
- Included paths: .
- Excluded paths: none
- Runtime or test status: not recorded

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 12 |
| Severity mix | critical: 1, high: 2, medium: 4, low: 5 |
| Confidence mix | high: 9, medium: 3 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

No explicit canonical threat-model summary was recorded.

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Cluster-privileged operator uses spec.workspaceName as an unauthenticated namespace selector: arbitrary namespace takeover and deletion](#finding-1) | critical | high | inline below |
| [Open WebUI exposed on a public Ingress with no signup/admin configuration: open self-registration and first-user admin claim](#finding-2) | high | medium | inline below |
| [Operator creates an unauthenticated Ingress exposing the raw Ollama API for every workspace](#finding-3) | high | medium | inline below |
| [Open WebUI container deliberately shipped with SecurityContext commented out while the sibling Ollama workload is hardened](#finding-4) | medium | high | inline below |
| [Default overlay deploys MySQL with credentials published in the repository (root 'password', app 'aichatworkspace') on EOL mysql:5.6](#finding-5) | medium | high | inline below |
| [Synchronous, timeout-less model pull inside Reconcile lets one CR starve reconciliation of every workspace](#finding-6) | medium | medium | inline below |
| [ClusterRole grants '\*' over namespaces, pods, pods/exec and all workload types cluster-wide (cluster-admin-equivalent operator)](#finding-7) | medium | high | inline below |
| [Workspace images selected from a mutable ConfigMap tag string interpolated without validation; shipped default is open-webui:main](#finding-8) | low | high | inline below |
| [spec.patterns interpolated unescaped into Ollama Modelfiles: SYSTEM triple-quote breakout allows arbitrary FROM/ADAPTER/TEMPLATE/PARAMETER directives](#finding-9) | low | high | inline below |
| [Build/devcontainer tooling downloaded and executed without checksum or signature verification](#finding-10) | low | high | inline below |
| [Unauthenticated pprof/expvar/statsviz debug endpoints always enabled on localhost:6060 of the cluster-privileged manager pod](#finding-11) | low | high | inline below |
| [hack/envoy-sidecar auth demo commits working credentials (Basic ollama:ollama, apr1 hash) and exposes Envoy admin on 0.0.0.0:9901](#finding-12) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Cluster-privileged operator uses spec.workspaceName as an unauthenticated namespace selector: arbitrary namespace takeover and deletion

| Field | Value |
| --- | --- |
| Severity | critical |
| Confidence | high |
| Confidence rationale | Source-verified end to end: no pattern/enum or webhook validation on workspaceName, no ownership comparison on any ensure path, ClusterRoleBinding confirms cluster-wide namespace delete, and the delete path runs purely from IsCreated + DeletionTimestamp + finalizer. |
| Category | broken-access-control |
| CWE | CWE-441, CWE-863, CWE-269 |
| Affected lines | internal/controller/aichatworkspace_controller.go:232-248, internal/controller/delete.go:30-51, internal/controller/namespace.go:36-64, internal/controller/handle_reconcile.go:44-134, internal/controller/ollama.go:51-132, api/v1alpha1/aichatworkspace_types.go:24-28, config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:56-62, config/rbac/role.yaml:13-24, config/rbac/role_binding.yaml:1-18 |

#### Summary

The reconciler treats spec.workspaceName as the literal target namespace for every operation and deletes it on CR deletion, without any ownership check, no validating webhook, and with cluster-wide wildcard RBAC. A user who can only create an AIChatWorkspace CR in their own namespace can make the operator create quotas, workloads, service accounts, PVCs and Ingresses inside any existing namespace (e.g. kube-system, default, another tenant's workspace) and then, by deleting the CR, cause deletion of that entire namespace.

#### Root Cause

spec.workspaceName is an attacker-chosen identifier used directly as the target namespace (and as the namespace delete argument) with no derivation from, or ownership check against, the requesting CR; namespaces cannot carry a namespaced-CR ownerReference and the ignored SetControllerReference error leaves only labels, which the delete path never inspects.

**Code evidence 1** — `internal/controller/aichatworkspace_controller.go:235-240`

```
namespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: instance.Spec.WorkspaceName}}
err := r.Delete(context.TODO(), namespace)
```

**Code evidence 2** — `internal/controller/namespace.go:41-63`

```
err := r.Get(context.TODO(), types.NamespacedName{Name: ns.Name}, found)
if err != nil && errors.IsNotFound(err) { ... Create ... }
// existing namespace: falls through and returns nil, nil with no ownership check
```

**Code evidence 3** — `config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:56-62`

```
workspaceName: ... XValidation rule="self == oldSelf" only; no pattern, no enum, no webhook (webhook server in cmd/main.go registers no handlers)
```

#### Validation

Attacker = namespaced CR author (editor role grants create/delete). Entry = CR create, then CR delete. Dataflow = spec.workspaceName -\> ensureNamespace (adopts existing namespace) -\> all ensure\* and Ollama model pull inside that namespace -\> IsCreated=true -\> finalizer -\> r.Delete(namespace). Control absent: no webhook, no label/owner check, no denylist. Counterevidence considered: workspaceName immutability (not a mitigation; chosen at create time), DNS-1123 namespace grammar (blocks syntax injection, not selection of existing names).

**Code evidence 1** — `internal/controller/delete.go:35-38`

```
if isCreated && pendingDeletion {
	if controllerutil.ContainsFinalizer(instance.aichatWorkspaceConfig, aichatWorkspaceFinalizerName) {
		if err = instance.r.deleteAIChatWorkspace(instance.ctx, instance.aichatWorkspaceConfig); err != nil {
```

**Code evidence 2** — `config/rbac/role.yaml:13-24`

```
resources: [namespaces, persistentvolumeclaims, pods, pods/exec, resourcequotas, serviceaccounts, services] verbs: ['*']  # ClusterRole bound cluster-wide by role_binding.yaml
```

**Code evidence 3** — `internal/controller/ollama.go:102-119`

```
ollamaServerURI := fmt.Sprintf("http://%s.%s.svc.cluster.local:%d", serviceName, instance.Spec.WorkspaceName, ollamaPort)
... ollama.PullModel(llm, ollamaServerURI) // attacker-chosen models pulled into the targeted namespace's Ollama
```

#### Dataflow

CR spec.workspaceName -\> ensureNamespace adopts existing namespace -\> quota/PVC/service-accounts/statefulset/deployment/ingress created or reused inside it -\> status.IsCreated set -\> CR deletion triggers deleteAIChatWorkspace -\> namespace deleted with all its workloads and PVCs.

#### Reachability

Reachable in any cluster with the shipped install; no additional prerequisites beyond the editor-role-equivalent CR rights. Target namespace must be a valid DNS-1123 label, which every real namespace is.

#### Severity

**Critical** — Namespaced, low-privilege attacker; cluster-wide destructive impact (cascading deletion of any namespace including kube-system, cross-tenant data destruction) plus pre-deletion pollution/DoS of system namespaces. High likelihood: the shipped aichatworkspace-editor-role grants exactly the namespaced CR rights assumed, and every gate (ensureNamespace existing-branch, IsCreated, finalizer) is satisfiable.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Derive the workspace namespace from the CR (e.g. \<cr-namespace\>--\<cr-name\>) and reject spec.workspaceName values that do not match; require an existing target namespace to carry an operator-written ownership label/annotation tied to this CR before any ensure\*/model-pull; before deleting a namespace verify the same ownership marker and denylist kube-system, kube-default, kube-public, kube-node-lease, and the operator namespace; stop ignoring the error from controllerutil.SetControllerReference; shrink the ClusterRole so namespace lifecycle rights cannot be exercised on arbitrary names.

<a id="finding-2"></a>

### [2] Open WebUI exposed on a public Ingress with no signup/admin configuration: open self-registration and first-user admin claim

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | medium |
| Confidence rationale | Source-verified that the operator ships no signup/admin/access restriction anywhere (exhaustive Env list in NewDeployment; git grep for ENABLE_SIGNUP/WEBUI_SECRET_KEY/auth finds nothing). The open-registration and first-user-admin behavior is upstream Open WebUI default, not this repo's code, which is why confidence is medium. |
| Category | insecure-defaults |
| CWE | CWE-1188, CWE-284 |
| Affected lines | internal/adapters/k8s/apps.go:76-97, internal/controller/handle_reconcile.go:119-126, internal/adapters/k8s/common.go:189-230 |

#### Summary

NewDeployment sets only OLLAMA_BASE_URL, OPENAI_API_BASE_URL, ENV=dev, WEBUI_NAME, KEY_FILE and the deployment is exposed on host \<workspace\>.\<defaultDomain\> with no auth. Open WebUI permits open registration by default and grants administrative control to the first registered account; nothing here disables signup, bootstraps an admin, or fronts the UI with authentication.

#### Root Cause

The generated deployment neither configures Open WebUI registration/admin defaults nor places any identity control in front of the public Ingress host.

**Code evidence 1** — `internal/adapters/k8s/apps.go:76-97`

```
Env: []v1.EnvVar{ {Name: "OLLAMA_BASE_URL"...}, {Name: "OPENAI_API_BASE_URL"...}, {Name: "ENV", Value: "dev"}, {Name: "WEBUI_NAME"...}, {Name: "KEY_FILE", Value: "/tmp/.webui_secret_key"} } // no ENABLE_SIGNUP, no admin bootstrap
```

**Code evidence 2** — `internal/controller/handle_reconcile.go:121-123`

```
openwebuiDNSName := setIngressDNSHost(config, aichat.Spec.WorkspaceName, constants.OpenwebuiName)
result, err = r.ensureIngress(ctx, aichat, k8s.NewIngress(aichat.Spec.WorkspaceName, constants.OpenwebuiName, openwebBackend, openwebuiDNSName, constants.OpenwebuiContainerPort))
```

#### Validation

Attacker = remote network user at \<workspace\>.\<defaultDomain\>. The Env slice in NewDeployment is exhaustive; no signup/access control variable is set, and the Ingress carries no auth annotations. KEY_FILE points at the ephemeral container layer (/tmp) while only /app/backend/data is persisted, so the session/JWT key is regenerated per pod restart.

**Code evidence 1** — `internal/adapters/k8s/apps.go:76-97`

```
Env: []v1.EnvVar{ {Name: "OLLAMA_BASE_URL"...}, {Name: "OPENAI_API_BASE_URL"...}, {Name: "ENV", Value: "dev"}, {Name: "WEBUI_NAME"...}, {Name: "KEY_FILE", Value: "/tmp/.webui_secret_key"} } // no ENABLE_SIGNUP, no admin bootstrap
```

#### Dataflow

Remote user -\> openwebui Ingress -\> registration endpoint -\> (first) admin account claim -\> inference and data access on the workspace backend.

#### Reachability

Same topology prerequisites as the Ollama exposure; additionally any in-cluster client can reach the openwebui ClusterIP service.

#### Severity

**High** — Unauthenticated network attacker can claim the admin account of a workspace instance or use its inference backend; impact is full workspace compromise. Likelihood depends on upstream default behavior plus deployment topology, so high/medium rather than critical.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Set ENABLE_SIGNUP=false (or an invite/SSO mechanism) and provide a fixed WEBUI_SECRET_KEY from a Secret (currently only an ephemeral /tmp KEY_FILE is set); front the openwebui Ingress with authentication by default; document admin bootstrap.

<a id="finding-3"></a>

### [3] Operator creates an unauthenticated Ingress exposing the raw Ollama API for every workspace

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | medium |
| Confidence rationale | The unauthenticated code path is source-verified and no default-installed control exists; the remaining uncertainty is deployment topology (whether an ingress controller is present) and product intent. |
| Category | missing-authentication |
| CWE | CWE-306 |
| Affected lines | internal/controller/handle_reconcile.go:128-134, internal/controller/handle_reconcile.go:150-163, internal/adapters/k8s/common.go:189-230, config/default/kustomization.yaml:30-39, config/default/system-configmap.yaml:6-8 |

#### Summary

handleReconcile always creates a second Ingress routing the Ollama service to host \<workspaceName\>-api.\<defaultDomain\> with no annotations, no auth backend, and no TLS. Ollama has no authentication, so anyone who can reach that host can pull arbitrary models into the 20Gi PVC, run unlimited inference, and create/delete/copy models. The only auth mechanisms in the repo (hack/envoy-sidecar, nginx basic-auth) are demos referenced by no kustomization.

#### Root Cause

The reconcile path unconditionally provisions a public route to an unauthenticated upstream service (Ollama) with no authentication layer and no default NetworkPolicy.

**Code evidence 1** — `internal/controller/handle_reconcile.go:129-131`

```
ollamaDNSName := setIngressDNSHost(config, aichat.Spec.WorkspaceName, constants.OllamaName)
result, err = r.ensureIngress(ctx, aichat, k8s.NewIngress(aichat.Spec.WorkspaceName, constants.OllamaName, ollamaBackend, ollamaDNSName, constants.OllamaPort))
```

**Code evidence 2** — `internal/adapters/k8s/common.go:204-227`

```
Spec: networkingv1.IngressSpec{Rules: ... HTTP: Paths: Path "/" Backend: Service ...} // no annotations, no auth, no TLS
```

#### Validation

Entry point: HTTP(S) to \<workspace\>-api.\<defaultDomain\>. No repo-shipped control authenticates the host: hack/envoy-sidecar and hack/ollama-ingress.yaml exist in no kustomization resources list; config/network-policy covers only operator metrics and is commented out of the default overlay. Ollama API capabilities (pull/create/copy/delete/generate) are exercised by the repo's own envoy README.

**Code evidence 1** — `internal/controller/handle_reconcile.go:155-158`

```
case "ollama": dnsName = fmt.Sprintf("%s-api.%s", workspace, config.DefaultDomain)
```

**Code evidence 2** — `config/default/system-configmap.yaml:6`

```
defaultDomain: "localtest.me"   // public wildcard DNS domain shipped as the default
```

#### Dataflow

Attacker request -\> Ingress host \<ws\>-api.\<domain\> -\> Ollama ClusterIP service :11434 -\> /api/pull, /api/generate, /api/create, /api/delete on the workspace instance.

#### Reachability

Default-reachable once an ingress controller and DNS exist; otherwise reachable by any in-cluster workload because no workspace NetworkPolicy is applied.

#### Severity

**High** — High impact (unmetered compute, PVC exhaustion, attacker-controlled model inventory) against a boundary the product itself creates; likelihood gated on an ingress controller + DNS (the repo quickstart installs ingress-nginx and ships the public wildcard domain localtest.me), and exposure may be an intended feature, so high rather than critical.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Do not create the Ollama API Ingress by default; when enabled, require an authentication mechanism (auth-proxy/OIDC or nginx auth-type with a real secret), pin allowed models, and ship a default NetworkPolicy denying cross-namespace access to workspace pods.

<a id="finding-4"></a>

### [4] Open WebUI container deliberately shipped with SecurityContext commented out while the sibling Ollama workload is hardened

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Literal commented-out source with the author's note; the hardened helpers exist in the same file and are applied only to the Ollama workload. |
| Category | container-hardening |
| CWE | CWE-250, CWE-1188 |
| Affected lines | internal/adapters/k8s/apps.go:63-75, internal/adapters/k8s/apps.go:182-219 |

#### Summary

NewDeployment comments out both the pod SecurityContext ('at the moment having issues getting open webui to run as non-root') and the container SecurityContext, so the open-webui container runs with the image default (root, all capabilities, no seccomp), while NewStatefulSet applies runAsNonRoot 10001, dropped capabilities, read-only root filesystem and RuntimeDefault seccomp to Ollama. A web-layer compromise therefore yields root in the container with write access to the workspace PVC and free egress to cluster ClusterIPs.

#### Root Cause

The deployment path knowingly skips the file's hardened security contexts for Open WebUI because of an unresolved non-root issue.

**Code evidence 1** — `internal/adapters/k8s/apps.go:69-75`

```
// at the moment having issues getting open webui to run as non-root
// SecurityContext: defaultPodSecurityContext(),
... // SecurityContext: defaultSecurityContext(),
```

**Code evidence 2** — `internal/adapters/k8s/apps.go:187-199`

```
SecurityContext: defaultPodSecurityContext(), // ollama sts: runAsNonRoot 10001, drop ALL, readOnlyRootFS, RuntimeDefault seccomp
```

#### Validation

Both workloads correctly set AutomountServiceAccountToken=false and carry no hostPath/hostNetwork/privileged; no NetworkPolicy is deployed for workspace namespaces, so a compromised container retains full cluster-internal egress (including MySQL).

**Code evidence 1** — `internal/adapters/k8s/apps.go:69-75`

```
// at the moment having issues getting open webui to run as non-root
// SecurityContext: defaultPodSecurityContext(),
... // SecurityContext: defaultSecurityContext(),
```

#### Dataflow

App compromise -\> root container -\> read/write of /app/backend/data PVC (chat data, webui.db), writable filesystem, raw network to all ClusterIP services.

#### Reachability

Second-stage impact; entry points are the exposed UI/API reported separately.

#### Severity

**Medium** — Defense-in-depth failure that materially widens a compromise whose entry points (open ingress exposure, open signup) are already reported; not itself remotely triggerable, so medium.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Run open-webui with runAsNonRoot, drop ALL capabilities and seccomp RuntimeDefault; split writable paths so readOnlyRootFilesystem is feasible; reuse defaultSecurityContext()/defaultPodSecurityContext() like the Ollama path.

<a id="finding-5"></a>

### [5] Default overlay deploys MySQL with credentials published in the repository (root 'password', app 'aichatworkspace') on EOL mysql:5.6

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | The values are literal committed stringData/env values applied by the default kustomize resources list; no override exists anywhere in the repo. |
| Category | hardcoded-credentials |
| CWE | CWE-798, CWE-259 |
| Affected lines | config/default/mysql-secret.yaml:1-7, config/default/mysql-deployment.yaml:16-31, config/default/kustomization.yaml:34-39 |

#### Summary

config/default/kustomization.yaml actively ships mysql-secret.yaml, mysql-deployment.yaml, mysql-storage.yaml and mysql-service.yaml. The secret holds password: 'password'; the deployment hardcodes MYSQL_PASSWORD 'aichatworkspace' and root from that secret, runs mysql:5.6 (EOL since Feb 2021) and exposes ClusterIP 3306 with no NetworkPolicy applied. No operator code consumes this database, so it is unused attack surface with public credentials.

#### Root Cause

Working database credentials were committed to a public repository and wired into the default install path instead of being generated or injected at deployment time.

**Code evidence 1** — `config/default/mysql-secret.yaml:6-7`

```
stringData:
  password: password
```

**Code evidence 2** — `config/default/kustomization.yaml:35-39`

```
- system-configmap.yaml
- mysql-secret.yaml
- mysql-storage.yaml
- mysql-deployment.yaml
- mysql-service.yaml   # applied by default; #- ../network-policy on line 34 remains commented out
```

#### Validation

git grep across config/, hack/, test/ and docs confirms these are the only real committed credential values in the default overlay; no Go code reads the database, so exposure is the MySQL instance itself. Counterevidence: values are obviously placeholder-like, and reachability requires in-cluster network position.

**Code evidence 1** — `config/default/mysql-deployment.yaml:17-30`

```
- image: mysql:5.6 ... MYSQL_USER aichatworkspace; MYSQL_PASSWORD value: "aichatworkspace"; MYSQL_ROOT_PASSWORD from secret mysql-secret key password
```

#### Dataflow

In-cluster client -\> mysql-service:3306 -\> authenticates with repo-published password/aichatworkspace -\> full access to the aichatworkspace database.

#### Reachability

Requires only cluster network access; no NetworkPolicy is enabled by default.

#### Severity

**Medium** — Any in-cluster workload can authenticate with repo-published credentials (full read/write, root) against every default installation that did not change them; impact bounded because the database is unused by the shipped code and requires cluster network position, so medium rather than high.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Remove the mysql-\* resources from the default overlay (or generate random credentials at install via sealed-secrets/external secret store); never ship working passwords; retire mysql:5.6.

<a id="finding-6"></a>

### [6] Synchronous, timeout-less model pull inside Reconcile lets one CR starve reconciliation of every workspace

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | medium |
| Confidence rationale | Source-verified that PullModel never honors a caller context and SetupWithManager sets no concurrency; exact stall duration depends on the Ollama registry response (large/stalled blob), which is attacker-selectable but not reproducible offline. |
| Category | denial-of-service |
| CWE | CWE-400 |
| Affected lines | internal/adapters/ollama/ollama.go:43-70, internal/controller/ollama.go:101-132, internal/controller/aichatworkspace_controller.go:133-145 |

#### Summary

spec.Models flows into ollama.PullModel, which uses context.Background() and http.DefaultClient with no timeout and blocks until the pull finishes. The controller runs with the default single worker (no MaxConcurrentReconciles), so a single attacker-chosen multi-GB or stalled model reference blocks status updates, workload creation, and deletion finalization for all other AIChatWorkspace objects.

#### Root Cause

Long-running external I/O driven by user input is performed inline in Reconcile with a detached, timeout-less context on a single-worker controller.

**Code evidence 1** — `internal/adapters/ollama/ollama.go:43-64`

```
func PullModel(modelName string, defaultBaseURL string) error { httpClient := http.DefaultClient ... ctx := context.Background() ... err = client.Pull(ctx, req, progressFunc) } // synchronous, no timeout, reconcile ctx never propagated
```

#### Validation

CreateFromModelFile, CopyModel, DeleteModel, ShowModel share the same context.Background() pattern; SetupWithManager registers with default concurrency, so controller-runtime serializes all AIChatWorkspace reconciles behind a blocked pull. Counterevidence: normal models and a responsive registry make stalls invisible; the abuse requires a large or stalled model reference.

**Code evidence 1** — `internal/controller/ollama.go:106-119`

```
for _, llm := range instance.Spec.Models { ok, err := ollama.DoesModelExist(llm, ollamaServerURI) ... err = ollama.PullModel(llm, ollamaServerURI)
```

#### Dataflow

spec.models -\> PullModel(context.Background(), no timeout) -\> single reconcile worker blocked -\> every workspace's reconcile queue stalls.

#### Reachability

Available to any CR author in any default install; no prerequisites beyond CR write access.

#### Severity

**Medium** — Availability loss of the whole operator product from a namespaced privilege, trivially repeatable; no confidentiality/integrity impact, hence medium despite high likelihood.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Propagate the reconcile context (with cancellation/timeout), validate model names against an allow-list/pattern, move pulls to async jobs (goroutine with ownerRef-tracked status or a Job resource), and raise MaxConcurrentReconciles with per-workspace isolation.

<a id="finding-7"></a>

### [7] ClusterRole grants '\*' over namespaces, pods, pods/exec and all workload types cluster-wide (cluster-admin-equivalent operator)

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | role.yaml and role_binding.yaml are shipped verbatim and generated directly from the markers in internal/controller/aichatworkspace_controller.go. |
| Category | privilege-management |
| CWE | CWE-269 |
| Affected lines | config/rbac/role.yaml:13-24, config/rbac/role_binding.yaml:1-18, internal/controller/aichatworkspace_controller.go:72-80 |

#### Summary

Generated from the controller markers, manager-role grants wildcard verbs over namespaces, pods, pods/exec, PVCs, services, service accounts, resource quotas, deployments, statefulsets, ingresses and httpscaledobjects, bound via ClusterRoleBinding. The reconciler never needs exec outside its own workspace pods (only du -sh on constant paths), so any input-driven flaw (as in the workspaceName finding) or manager-pod compromise escalates to pod creation and exec across kube-system and every tenant.

#### Root Cause

Cluster-wide wildcard RBAC far exceeds the demonstrated needs of the reconcile path, and is the amplifier that turns CR input bugs into cluster compromise.

**Code evidence 1** — `config/rbac/role.yaml:13-24`

```
resources: [namespaces, persistentvolumeclaims, pods, pods/exec, resourcequotas, serviceaccounts, services] verbs: ['*'] (+ deployments, statefulsets, ingresses, httpscaledobjects '*')
```

**Code evidence 2** — `internal/adapters/k8s/common.go:304-343`

```
ExecuteRemoteCommand ... SubResource("exec") ... Command: []string{"/bin/sh", "-c", command} // only ever used for du -sh <constant path>, yet granted cluster-wide
```

#### Validation

Cross-checked role.yaml against every API call the reconciler makes: the only execs are fixed du commands; namespace writes are only used for the (already-faulty) workspaceName selection. Counterevidence: namespace creation is inherent to the product design; the manager runs non-root with dropped capabilities.

**Code evidence 1** — `config/rbac/role.yaml:13-24`

```
resources: [namespaces, persistentvolumeclaims, pods, pods/exec, resourcequotas, serviceaccounts, services] verbs: ['*'] (+ deployments, statefulsets, ingresses, httpscaledobjects '*')
```

#### Dataflow

Manager identity + '\*' on pods/pods/exec/namespaces cluster-wide -\> pod creation or exec in kube-system and every tenant namespace -\> cluster takeover.

#### Reachability

Amplification surface; direct use requires the manager identity or an operator input-handling flaw such as the workspaceName takeover reported in this scan.

#### Severity

**Medium** — Not directly exploitable by a lower-privileged principal; severity comes from amplification of the CR-driven flaws and of manager-pod compromise. The manager pod itself is hardened (non-root, dropped caps) and some breadth (namespace create) is intrinsic to the product, so medium.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Restrict verbs to what is used; scope pods/exec to workspace-owned pods via namespaced Roles; move namespace rights behind the ownership validation; prefer namespaced per-workspace Roles over the single cluster-wide wildcard.

<a id="finding-8"></a>

### [8] Workspace images selected from a mutable ConfigMap tag string interpolated without validation; shipped default is open-webui:main

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Sprintf interpolation of unvalidated ConfigMap values and the shipped ':main' default are both verbatim; no sha256 pin exists anywhere in the repo. |
| Category | supply-chain |
| CWE | CWE-1104, CWE-1188 |
| Affected lines | config/default/system-configmap.yaml:6-8, internal/adapters/k8s/apps.go:49, internal/adapters/k8s/apps.go:154, internal/config/config.go:37-69 |

#### Summary

GetConfig reads openwebuiImageTag/ollamaImageTag/defaultDomain from the operator ConfigMap and apps.go composes the image as fmt.Sprintf("%s:%s", name, tag) with zero validation and no digest. system-configmap.yaml ships openwebuiImageTag: "main", so every recreated workspace pod silently tracks a mutable upstream tag; a configmap-writer can silently select any image version for all new workspaces, and defaultDomain from the same ConfigMap flows unvalidated into every Ingress host.

#### Root Cause

Image identity is delegated to an unvalidated mutable string sourced from a ConfigMap, with no digest pinning or format validation.

**Code evidence 1** — `config/default/system-configmap.yaml:6-8`

```
defaultDomain: "localtest.me"
openwebuiImageTag: "main"
ollamaImageTag: "0.5.4"
```

**Code evidence 2** — `internal/adapters/k8s/apps.go:49`

```
containerImage := fmt.Sprintf("%s:%s", constants.OpenwebuiContainerImageName, openwebuiContainerImageTag)
```

#### Validation

Ollama is pinned to 0.5.4 (still a mutable tag, but versioned); Open WebUI's 'main' is a moving branch tag. The same ConfigMap's defaultDomain reaches Ingress hosts unvalidated. Counterevidence: the shipped manager ClusterRole lacks configmaps read permission (functional gap), so any working deployment adds RBAC of unknown scope.

**Code evidence 1** — `config/default/system-configmap.yaml:6-8`

```
defaultDomain: "localtest.me"
openwebuiImageTag: "main"
ollamaImageTag: "0.5.4"
```

#### Dataflow

ConfigMap tag -\> Sprintf image reference -\> every new/recreated workspace pod runs the substituted image.

#### Reachability

Requires upstream tag substitution or operator-namespace write access; affects new workspaces only.

#### Severity

**Low** — Supply-chain exposure contingent on an upstream 'main' re-push or on write access to the operator-namespace ConfigMap (already a privileged position); tag grammar bounds it to version selection, not registry redirection.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Pin images by digest (name:tag@sha256:...), validate tags against ^\[a-zA-Z0-9_.-\]+$ when reading the ConfigMap, and scope the config ConfigMap with tight RBAC.

<a id="finding-9"></a>

### [9] spec.patterns interpolated unescaped into Ollama Modelfiles: SYSTEM triple-quote breakout allows arbitrary FROM/ADAPTER/TEMPLATE/PARAMETER directives

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The sprintf with no escaping is verbatim in prompt(); CRD carries no pattern constraint on models/patterns; the embedded allow-list is confirmed unreferenced. |
| Category | injection |
| CWE | CWE-74 |
| Affected lines | internal/adapters/ai/modelfiles/modelfiles.go:38-51, internal/adapters/ollama/ollama.go:304-316, api/v1alpha1/aichatworkspace_types.go:33-38, config/crd/bases/apps.aichatworkspaces.io_aichatworkspaces.yaml:42-53 |

#### Summary

GetSystemPromptPattern builds the Modelfile with fmt.Sprintf over raw spec.models/spec.patterns values: a pattern containing """ followed by a newline terminates the SYSTEM block and injects arbitrary Modelfile directives, which the operator submits to the workspace Ollama /api/create. The embedded fabric pattern map that could serve as an allow-list is dead code (variables.go embeds are never referenced).

#### Root Cause

Free-form CR strings are concatenated into the Modelfile instruction language instead of being looked up against the embedded, bounded pattern set.

**Code evidence 1** — `internal/adapters/ai/modelfiles/modelfiles.go:38-51`

```
var promptTemplate = "\nFROM %s ... SYSTEM \"\"\"\n%s\"\"\""; return fmt.Sprintf(promptTemplate, model, pattern) // no escaping of quotes/newlines
```

#### Validation

Model pull/create requests are JSON (official ollama/api client), so this is directive-language injection into the Modelfile body, not URL/path injection. Ollama's own model-name validation bounds the FROM interpolation, and worst-case server-side consequences inside Ollama's model processor are not established by this repository.

**Code evidence 1** — `internal/adapters/ollama/ollama.go:304-311`

```
for _, pattern := range patterns { createModelName := fmt.Sprintf("%s-%s", modelName, pattern); modelfile := modelfiles.GetSystemPromptPattern(modelName, pattern); err = client.Create(ctx, &ollama.CreateRequest{Model: createModelName, Modelfile: modelfile}, progressFunc)
```

#### Dataflow

spec.patterns -\> prompt() sprintf -\> /api/create Modelfile -\> injected directives executed by the workspace Ollama.

#### Reachability

Any workspace whose CR the attacker controls; delivered cross-tenant only in combination with the workspaceName takeover finding.

#### Severity

**Low** — Within a workspace the creator already controls the same Ollama API (and can name any base model via spec.models), so the injection adds no privilege there; residual impact is integrity of operator-generated 'pattern models', poisoned chat templates shaping that workspace's end users, and remote-content fetch via ADAPTER. Downgraded to low as a same-tenant control gap; cross-tenant delivery of this primitive is captured in the workspaceName finding.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Resolve patterns by key against the embedded pattern map (reject unknown keys), validate model names against the Ollama name grammar in the CRD, and escape or reject quote/newline sequences before Modelfile interpolation.

<a id="finding-10"></a>

### [10] Build/devcontainer tooling downloaded and executed without checksum or signature verification

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The download-and-execute commands are verbatim; all other tool installs use pinned go install through checksummed proxies (verified), which bounds the finding to these specific paths. |
| Category | supply-chain |
| CWE | CWE-494 |
| Affected lines | Makefile:153-159, Makefile:161-164, .devcontainer/post-install.sh:4-15 |

#### Summary

The Makefile helm target pipes curl -s https://get.helm.sh/... | tar into LOCALBIN and executes it without checksum or signature verification; helmify is installed unpinned (@latest, with the target duplicated); .devcontainer/post-install.sh downloads 'latest' kind and kubebuilder and a kubectl binary with no checksum. A substituted download yields code execution on maintainer machines feeding the release pipeline.

#### Root Cause

Trusted binaries are fetched over HTTPS but executed with no integrity verification, and one path tracks unpinned 'latest'.

**Code evidence 1** — `Makefile:153-159`

```
HELM_INSTALLER ?= "https://get.helm.sh/helm-v3.10.1-$(OSARCH).tar.gz" ... cd $(LOCALBIN) && curl -s $(HELM_INSTALLER) | tar -xzf - -C $(LOCALBIN)
```

#### Validation

Checked all other tool installs (kustomize, controller-gen, setup-envtest, golangci-lint) use pinned go install with module checksums; workflows are fully commented out (no active CI surface today). Counterevidence: over HTTPS and pinned v3.10.1 for helm, limiting exposure to endpoint compromise or network position.

**Code evidence 1** — `.devcontainer/post-install.sh:4-13`

```
curl -Lo ./kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64 ... curl -L -o kubebuilder https://go.kubebuilder.io/dl/latest/linux/amd64 ... curl -LO https://dl.k8s.io/release/$KUBECTL_VERSION/... without checksums
```

#### Dataflow

curl | tar -\> LOCALBIN -\> executed by helm-build/helm-build releases on contributor machines.

#### Reachability

Developer/maintainer context only; no runtime cluster impact.

#### Severity

**Low** — Maintainer/CI network-position attacker only; the shipped Dockerfile is clean (two-stage, distroless nonroot, USER 65532, no curl|sh) and all GitHub workflow files are fully commented out, so no release path currently executes these targets unattended.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Verify SHA-256/provenance for helm/kind/kubebuilder/kubectl downloads, pin helmify, and prefer go-install or cosign-verified artifacts.

<a id="finding-11"></a>

### [11] Unauthenticated pprof/expvar/statsviz debug endpoints always enabled on localhost:6060 of the cluster-privileged manager pod

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Unconditional startup of the mux is verbatim; the contrast with the authn/authz-filtered metrics server shows the difference is deliberate; bind-address scope (pod netns only) also verified. |
| Category | information-disclosure |
| CWE | CWE-489, CWE-200 |
| Affected lines | cmd/main.go:106-124 |

#### Summary

main.go unconditionally serves pprof (including /debug/pprof/profile and heap/goroutine dumps), expvar and statsviz on localhost:6060; the ListenAndServe error is ignored. Inside a pod, localhost means any same-pod network-namespace peer (sidecar, ephemeral debug container) and anyone with pods/exec or port-forward rights reaches these listeners. Heap dumps expose the mounted ServiceAccount token and in-memory config, and CPU profiles are a cheap DoS against the 500m-limited single replica.

#### Root Cause

Production manager always compiles in and starts an unauthenticated profiling/introspection endpoint on the pod loopback.

**Code evidence 1** — `cmd/main.go:113-124`

```
go func() { mux := http.NewServeMux(); mux.HandleFunc("/debug/pprof/", pprof.Index); ... statsviz.Register(mux); http.ListenAndServe("localhost:6060", mux) }() // no auth, error ignored
```

#### Validation

Metrics (:8443 via patch) uses filters.WithAuthenticationAndAuthorization and health probes are pings only, so :6060 is the sole unauthenticated manager listener. Reachability is bounded to the pod network namespace; no hostNetwork.

**Code evidence 1** — `cmd/main.go:113-124`

```
go func() { mux := http.NewServeMux(); mux.HandleFunc("/debug/pprof/", pprof.Index); ... statsviz.Register(mux); http.ListenAndServe("localhost:6060", mux) }() // no auth, error ignored
```

#### Dataflow

loopback :6060 -\> /debug/pprof/heap (memory incl. SA token), /profile (CPU DoS), /debug/vars (flags/build), statsviz live telemetry.

#### Reachability

Pod-local only by default; hack/envoy-sidecar demonstrates sidecar injection into workspace pods as a supported pattern.

#### Severity

**Low** — No remote or cross-pod reachability in the default single-container distroless pod; exploitation requires an existing same-pod foothold or exec/port-forward rights (a position from which the SA token is often already readable). Amplifier rather than standalone breach: the SA it protects is cluster-admin-adjacent.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Remove the debug mux or gate it behind a flag/build tag defaulting off; check the ListenAndServe error; if profiling is needed, use ephemeral debug containers or front it with authentication.

<a id="finding-12"></a>

### [12] hack/envoy-sidecar auth demo commits working credentials (Basic ollama:ollama, apr1 hash) and exposes Envoy admin on 0.0.0.0:9901

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Literal committed values verified at the cited lines. |
| Category | hardcoded-credentials |
| CWE | CWE-798, CWE-327 |
| Affected lines | hack/envoy-sidecar/envoy-config.yaml:36, hack/envoy-sidecar/envoy-config.yaml:60-65, hack/envoy-sidecar/README.md:8-9, hack/envoy-sidecar/ollama-ingress.yaml:7 |

#### Summary

The repo's own 'secured Ollama' recipe routes on a literal Authorization header match of Basic b2xsYW1hOm9sbGFtYQ== (ollama:ollama), documents an htpasswd line with an Apache MD5-crypt ($apr1$) hash and plaintext --user ollama:ollama, and binds Envoy admin to 0.0.0.0:9901 with the port exported as a Service port. Anyone who can read the public repo authenticates against any unchanged adoption; admin port allows config dumps/stats in-cluster.

#### Root Cause

Static, published credential material is the entire access-control story of the shipped secure-path demo, plus an in-cluster admin listener.

**Code evidence 1** — `hack/envoy-sidecar/envoy-config.yaml:36`

```
exact_match: "Basic b2xsYW1hOm9sbGFtYQ=="  // base64 of ollama:ollama
```

**Code evidence 2** — `hack/envoy-sidecar/README.md:8-9`

```
# echo  "ollama:$apr1$i9ygHJeq$DtM1NF4LbsHMeo3WBMQKV0" > auth
kubectl create secret generic basic-auth --from-file=auth -n envoy-test
```

#### Validation

hack/envoy-sidecar has no kustomization and appears in no resources list (verified), bounding this to opt-in usage; the nginx fallback (ollama-ingress.yaml) references the same repo-generated secret pattern.

**Code evidence 1** — `hack/envoy-sidecar/README.md:8-9`

```
# echo  "ollama:$apr1$i9ygHJeq$DtM1NF4LbsHMeo3WBMQKV0" > auth
kubectl create secret generic basic-auth --from-file=auth -n envoy-test
```

#### Dataflow

Repo credential -\> Authorization header -\> envoy-gated Ollama API; Envoy admin :9901 reachable from in-cluster.

#### Reachability

Only where the demo manifests were applied; not default-deployed.

#### Severity

**Low** — Optional demo material outside every kustomization (envoy-test/localtest.me), so no default-install impact; reported because it is the project's shipped answer to protecting the Ollama API and users are likely to copy it verbatim.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Replace committed hashes/credentials with placeholders and generate-per-install instructions; use bcrypt htpasswd; bind Envoy admin to 127.0.0.1 and remove it from the Service.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| Operator control plane and reconcile path (cmd/, internal/controller/, internal/adapters/) | not recorded | Reported | No additional canonical notes were recorded. |
| AIChatWorkspace API types and CRD validation rules (api/v1alpha1/, config/crd/) | not recorded | Reported | No additional canonical notes were recorded. |
| RBAC and default deploy manifests (config/rbac/, config/default/, config/manager/) | not recorded | Reported | No additional canonical notes were recorded. |
| Generated workspace workloads: Ollama StatefulSet, Open WebUI Deployment, Services, Ingresses | not recorded | Reported | No additional canonical notes were recorded. |
| hack/ example manifests, envoy sidecar demo, and scale-to-zero material | not recorded | Reported | No additional canonical notes were recorded. |
| Dockerfile, Makefile, from-scratch.sh, .devcontainer/, .github/workflows/ | not recorded | Reported | No additional canonical notes were recorded. |
| Metrics, health probes, and webhook listener configuration | not recorded | Rejected | No additional canonical notes were recorded. |
| pods/exec command construction, Ollama URI/SSRF, label and ingress-host syntax injection | not recorded | Rejected | No additional canonical notes were recorded. |
| Embedded fabric prompt markdown (internal/adapters/ai/modelfiles/files/) | not recorded | No issue found | No additional canonical notes were recorded. |

## Open Questions And Follow Up

- Whether the operator-authored Ingress exposing the raw Ollama API is an intended product feature; README/notes do not state, and the only auth mechanisms (hack/envoy-sidecar, nginx basic-auth) are demos outside every kustomization.
- Whether Ollama's Modelfile processor (external binary, image tag chosen from a ConfigMap) turns injected ADAPTER/TEMPLATE directives into code execution; the injection primitive itself is established in this repository, its server-side amplification is not.
- Whether deployed installations add a validating webhook, policy engine, or derived-namespace convention outside this repository; the shipped manifests register none.
- The shipped manager ClusterRole grants no configmaps permission even though internal/config/config.go must read the operator ConfigMap; working deployments must add RBAC of unknown extent.
