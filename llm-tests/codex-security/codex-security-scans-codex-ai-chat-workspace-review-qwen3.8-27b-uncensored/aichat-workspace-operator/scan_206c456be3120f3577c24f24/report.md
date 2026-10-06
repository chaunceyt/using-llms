# Security Review: aichat-workspace-operator

## Scope

Whole repository (Git worktree at main).

- Scan mode: repository
- Target kind: git_worktree
- Target ID: target_6e8869f9a0f00c035e26d2cbacecb3a04f25a8f6
- Revision: 6bca2ac4706df6e392d748308c1a163d30276b2e
- Snapshot digest: codex-security-snapshot/v1:sha256:1239d7cf703fd6ff3b727dd36fd601683b6d42567007d586fd21b865a7f907c7
- Inventory strategy: repository
- Included paths: .
- Excluded paths: none
- Runtime or test status: not recorded

### Scan Summary

| Field | Value |
| --- | --- |
| Scan outcome | completed |
| Reportable findings | 4 |
| Severity mix | high: 1, medium: 1, low: 2 |
| Confidence mix | high: 2, medium: 2 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

A Kubebuilder/operator written in Go that, for each AIChatWorkspace custom resource, provisions a dedicated namespace and runs Open WebUI (web app) plus Ollama (LLM API) with public DNS-hosted Ingresses. Primary trust boundaries: the K8s API / CR author, the cluster network, and the public internet reaching the Ingress-fronted Ollama and Open WebUI services. Protected assets are each tenant's workspace (models, chat data on PVCs, cluster resources).

### Assets

- Per-workspace Ollama LLM API and stored models
- Per-workspace Open WebUI application, accounts, and chat data (PVC)
- Cluster compute/storage (ResourceQuota-limited namespaces)
- Shared operator service account and cluster RBAC

### Trust Boundaries

- Kubernetes API server / AIChatWorkspace CR authors
- Cluster network (svc.cluster.local)
- Public internet -\> Ingress controller -\> workspace subdomains

### Attacker Capabilities

- A user authorized to create/update AIChatWorkspace custom resources (workspace owner)
- An unauthenticated internet user who can resolve/brute-force a workspace subdomain

### Security Objectives

- Isolation of tenant workspaces
- Authentication of external LLM/UI access
- Encryption of credentials in transit
- Bounded operator privileges

### Assumptions

- The cluster ingress controller is exposed to the internet (product intent: external LLM-as-a-Service).
- Each workspace is isolated in its own namespace/Ollama instance (no cross-tenant network path in source).

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Unauthenticated Ollama LLM API is publicly exposed via Ingress](#finding-1) | high | medium | inline below |
| [Open WebUI is exposed publicly over cleartext HTTP with open self-registration](#finding-2) | medium | high | inline below |
| [pprof, expvar, and statsviz debug endpoints are enabled in production](#finding-3) | low | high | inline below |
| [Unvalidated WorkspaceName is reflected into public DNS subdomains and K8s object names](#finding-4) | low | medium | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Unauthenticated Ollama LLM API is publicly exposed via Ingress

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | medium |
| Confidence rationale | Source unambiguously shows a public-host Ingress with no TLS and no auth in front of the Ollama API. Severity assumes the cluster ingress controller is internet-facing, which is the product's stated design but not verifiable from the repository. |
| Category | broken-access-control |
| CWE | CWE-306, CWE-319 |
| Affected lines | internal/adapters/k8s/common.go:189-231, internal/controller/handle_reconcile.go:128-134, internal/adapters/k8s/apps.go:182-219, internal/controller/handle_reconcile.go:153-162 |

#### Summary

The operator creates an Ingress that routes the full Ollama API (port 11434) to a public DNS host (\<workspace\>-api.\<defaultDomain\>) with no TLS and no authentication. Ollama provides no auth by default and none is configured, so any internet user can run inference, pull, create, and delete models in the workspace.

#### Root Cause

The reconciliation path publishes the Ollama API on a public host without any transport or identity layer. Ollama's own API is unauthenticated, so the only barrier is network reachability.

**Ingress built with a Host, no TLS block, no auth annotations** — `internal/adapters/k8s/common.go:189-231`

NewIngress returns an IngressSpec with a single Rule (Host set to the caller-supplied hostname) and no Spec.TLS and no authentication annotations, so the backend is served over HTTP with no access control.

```go
func NewIngress(workspacename, workload, backendName, hostname string, backendPort int32) *networkingv1.Ingress {
	pathType := networkingv1.PathTypePrefix
	return &networkingv1.Ingress{...
		Spec: networkingv1.IngressSpec{
			Rules: []networkingv1.IngressRule{
				{ Host: hostname,
				  IngressRuleValue: ... Path: "/" ... }
			}
		},
	}
```

**Ollama Ingress created and reconciled to a public host** — `internal/controller/handle_reconcile.go:128-134`

handleReconcile unconditionally creates the Ollama Ingress pointing at the Ollama service port with a public host derived from the workspace name.

```go
ollamaBackend := getName(aichat.Spec.WorkspaceName, constants.OllamaName)
ollamaDNSName := setIngressDNSHost(config, aichat.Spec.WorkspaceName, constants.OllamaName)
result, err = r.ensureIngress(ctx, aichat, k8s.NewIngress(aichat.Spec.WorkspaceName, constants.OllamaName, ollamaBackend, ollamaDNSName, constants.OllamaPort))
```

**Ollama StatefulSet has no auth; debug enabled** — `internal/adapters/k8s/apps.go:189-219`

The Ollama workload is deployed with no authentication mechanism (no OLLAMA_API_KEY, no proxy, no sidecar) and OLLAMA_DEBUG=1.

```go
return &appsv1.StatefulSet{ ... Env: []v1.EnvVar{{Name:"OLLAMA_DEBUG",Value:"1"}} ... }
```

#### Validation

Confirmed by reading NewIngress (no TLS/auth), handleReconcile (unconditional public-host Ollama Ingress), the Ollama StatefulSet (no auth), and the Ollama adapter (plain HTTP client to the internal service).

Validation method: static

- **Status:** validated

Counterevidence and remaining uncertainty:
- If the ingress controller is not internet-exposed, or a cluster-level auth proxy (e.g. the envoy-sidecar in hack/) is deployed, the exposure is mitigated.
- Ollama is bound to the internal service; only the Ingress provides the external path.

Limitations:
- External exposure of the ingress controller and DNS are deployment properties not present in the repository.

#### Dataflow

Public internet -\> ingress controller (host \<w\>-api.\<domain\>) -\> Ollama service :11434 -\> model store on PVC.

- **Source:** Unauthenticated internet client

- **Sink:** Ollama HTTP API (/api/pull, /api/create, /api/delete, /api/generate, /api/tags)

- **Outcome:** Arbitrary model pull/create/delete and inference, and resource exhaustion, within the target workspace.

#### Reachability

Reachability was not recorded beyond the canonical finding summary and affected locations.

#### Severity

**High** — Unauthenticated, remote, internet-reachable control plane for a stateful service: enables model tampering/destruction, resource-exhaustion DoS against the quota-limited workspace, and data access. Per-workspace blast radius (tenants are namespace-isolated).

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Add authentication for the Ollama API (basic/JWT/OIDC via an ingress annotation, an authenticating proxy/sidecar, or OLLAMA_API_KEY if available) and enable TLS on the ingress. Restrict the Ollama Ingress to a private/vpc path or require a token before exposing it publicly.

<a id="finding-2"></a>

### [2] Open WebUI is exposed publicly over cleartext HTTP with open self-registration

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | The Ingress has no TLS and no auth and routes the full web app; the Deployment sets no auth/registration restrictions (KEY_FILE points at ephemeral /tmp). External exposure is an implied deployment property. |
| Category | security-misconfiguration |
| CWE | CWE-319, CWE-311, CWE-306 |
| Affected lines | internal/adapters/k8s/common.go:189-231, internal/controller/handle_reconcile.go:119-126, internal/adapters/k8s/apps.go:71-97 |

#### Summary

The operator publishes the Open WebUI application on a public host (\<workspace\>.\<defaultDomain\>) over plain HTTP (no TLS on the Ingress) and does not configure authentication. Open WebUI's default allows open self-registration, so any internet user can create an account and use/abuse the workspace; credentials are transmitted in cleartext.

#### Root Cause

The web UI is published to a public host over HTTP with no identity layer and no registration restriction, so it is trivially reachable and account-creatable by anyone.

**Open WebUI Ingress created with a public host, no TLS** — `internal/controller/handle_reconcile.go:119-126`

Creates the public-host Ingress for the Open WebUI service with the same unauthenticated, non-TLS NewIngress builder.

```go
openwebuiDNSName := setIngressDNSHost(config, aichat.Spec.WorkspaceName, constants.OpenwebuiName)
result, err = r.ensureIngress(ctx, aichat, k8s.NewIngress(aichat.Spec.WorkspaceName, constants.OpenwebuiName, openwebBackend, openwebuiDNSName, constants.OpenwebuiContainerPort))
```

**Open WebUI Deployment: no auth config, ephemeral secret key** — `internal/adapters/k8s/apps.go:71-97`

The web UI runs without any authentication/registration restriction and keeps its session/JWT secret in ephemeral /tmp, and has no SecurityContext (runs as root).

```go
// at the moment having issues getting open webui to run as non-root
// SecurityContext: defaultPodSecurityContext(),
... {Name:"KEY_FILE",Value:"/tmp/.webui_secret_key"}
```

#### Validation

NewIngress adds no TLS/auth; handleReconcile always creates the public Open WebUI Ingress; NewDeployment sets no auth or registration controls.

Validation method: static

- **Status:** validated

Counterevidence and remaining uncertainty:
- Open WebUI does provide its own register/login (just open by default).
- External exposure depends on the cluster's ingress/DNS configuration.

Limitations:
- Whether registration is truly open at runtime and whether the ingress is public are deployment properties.

#### Dataflow

Internet -\> ingress host \<w\>.\<domain\> -\> Open WebUI :8080 -\> (register) -\> LLM via internal Ollama.

- **Source:** Unauthenticated internet user

- **Sink:** Open WebUI registration/login and backend LLM access

- **Outcome:** Unvetted accounts on the tenant's UI and cleartext credential exposure; account takeover of any weak/self-registered credential.

#### Reachability

Reachability was not recorded beyond the canonical finding summary and affected locations.

#### Severity

**Medium** — Public, unauthenticated account creation plus cleartext credentials against a tenant's LLM UI; lower than the Ollama finding because Open WebUI ships its own (open by default) auth layer.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Enable TLS on the ingress, disable Open WebUI's open self-registration (require an IdP or an admin-managed bootstrap account), and configure a stable persisted session secret; run the container with a SecurityContext (non-root).

<a id="finding-3"></a>

### [3] pprof, expvar, and statsviz debug endpoints are enabled in production

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | The code path is unconditional and unconditional in main(); the exposure scope (localhost) is directly readable. |
| Category | insecure-configuration |
| CWE | CWE-489, CWE-215, CWE-200 |
| Affected lines | cmd/main.go:113-123 |

#### Summary

cmd/main.go unconditionally starts an HTTP server on localhost:6060 exposing /debug/pprof/\* (including /debug/pprof/cmdline), expvar (/debug/vars/), and statsviz. These diagnostic surfaces leak process/runtime information and the full command line and are left enabled with no flag or authentication.

#### Root Cause

Diagnostic/debug endpoints are hardwired on and not gated behind a flag or access control.

**Unauthenticated debug server on :6060** — `cmd/main.go:113-123`

A goroutine registers pprof (incl. cmdline), expvar, and statsviz handlers and listens with no authentication.

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

#### Validation

Read cmd/main.go; the server is always started and binds to localhost:6060 with no auth.

Validation method: static

- **Status:** validated

Counterevidence and remaining uncertainty:
- Binds to localhost, not a public interface, so it is not directly internet-reachable without another access path.

Limitations:
- Blast radius depends on what else can reach port 6060 in the deployment.

#### Dataflow

Local/port-forwarded client -\> :6060 -\> /debug/pprof/cmdline, /debug/vars/, statsviz runtime data.

- **Source:** Same-node / co-located attacker or kubectl port-forward

- **Sink:** pprof/expvar/statsviz handlers

- **Outcome:** Disclosure of the operator's command line, environment-derived values, and live runtime/goroutine state.

#### Reachability

Reachability was not recorded beyond the canonical finding summary and affected locations.

#### Severity

**Low** — Information disclosure limited to the pod/node (bound to localhost); exploitable only by an attacker already able to reach the port (e.g., same-node, port-forward, or co-located container).

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Gate the debug server behind a flag (disabled by default) or remove statsviz/expvar in production, and/or restrict it to the pod's loopback with an auth filter and document that it must not be exposed via a Service.

<a id="finding-4"></a>

### [4] Unvalidated WorkspaceName is reflected into public DNS subdomains and K8s object names

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | Source shows no validation and direct reflection into the host; the concrete impact (collision with a real service under the default domain) depends on the chosen defaultDomain and tenant names. |
| Category | improper-input-validation |
| CWE | CWE-20 |
| Affected lines | api/v1alpha1/aichatworkspace_types.go:24-28, internal/controller/handle_reconcile.go:153-162, internal/controller/handle_reconcile.go:119-134 |

#### Summary

Spec.WorkspaceName has only an immutability rule (no pattern/length validation) and is reflected into the public ingress host (\<workspace\>.\<defaultDomain\> / \<workspace\>-api.\<defaultDomain\>) and into namespace/object names and labels. A CR author can thus claim an arbitrary single-label subdomain under the shared default domain (subdomain squatting/collision) and shape object names.

#### Root Cause

WorkspaceName is an unvalidated, attacker-selectable string that is interpolated into externally-observable DNS hosts and K8s object/label names without a per-tenant subdomain namespace or reserved-word protection.

**WorkspaceName spec field: only immutability, no pattern/length** — `api/v1alpha1/aichatworkspace_types.go:24-28`

The only validation on WorkspaceName is that it is immutable; there is no maxLength or pattern constraint.

```go
type AIChatWorkspaceSpec struct {
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="WorkspaceName is immutable"
	WorkspaceName string `json:"workspaceName"`
```

**WorkspaceName reflected into the public ingress host** — `internal/controller/handle_reconcile.go:153-162`

The unvalidated workspace string is concatenated directly into a public DNS hostname under the shared defaultDomain.

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
```

#### Validation

Confirmed no pattern/length validation in the CRD type and direct reflection in setIngressDNSHost.

Validation method: static

- **Status:** validated

Counterevidence and remaining uncertainty:
- Because workspaceName is also the namespace name, the API server enforces DNS-1123 label rules, bounding the set of usable values to a single lowercase label.
- Impact requires the defaultDomain to host other services a tenant could impersonate.

Limitations:
- Concrete impact depends on the configured defaultDomain and multi-tenant usage.

#### Dataflow

CR author sets workspaceName -\> setIngressDNSHost -\> Ingress Host under shared defaultDomain.

- **Source:** AIChatWorkspace CR author (workspaceName)

- **Sink:** Ingress Host (\<w\>\[ -api\].\<defaultDomain\>) and namespace/object names

- **Outcome:** Subdomain squatting/collision under the operator's shared domain and control of object naming.

#### Reachability

Reachability was not recorded beyond the canonical finding summary and affected locations.

#### Severity

**Low** — Low impact: Kubernetes name rules bound most abuse and tenants are namespace-isolated, but the shared-domain subdomain is attacker-selectable, enabling collision/impersonation under the operator's default domain.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Add +kubebuilder validation:Pattern (DNS-1123 label) and maxLength to WorkspaceName, and/or place each workspace under its own subdomain (e.g. \<w\>.workspaces.\<domain\>) with reserved names blocked and collision checks.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| Public Ingress exposure of Ollama and Open WebUI | not recorded | Reported | Unauthenticated, non-TLS public hosts. |
| Ollama HTTP adapter (pull/create/list via internal service) | not recorded | No issue found | Client connects only to the workspace's own internal Ollama service; model/pattern values are used by design (system prompts / model names). No SSRF to arbitrary hosts. |
| ExecuteRemoteCommand (pod exec /bin/sh -c) | not recorded | No issue found | Command string is a hardcoded constant (du -sh \<mount\>) today; latent command-exec primitive noted for future features. |
| Modelfile / system-prompt template generation | not recorded | No issue found | Patterns/model names are interpolated into Ollama modelfiles by design (the product feature); embedded data is static (go:embed). No code execution or traversal. |
| Operator RBAC breadth (verbs=\* on core + networking resources) | not recorded | No issue found | Operator SA has very broad RBAC incl. pods/exec; large blast radius if the operator is compromised, but not directly attacker-reachable. See report notes. |
| Secrets / credential handling | not recorded | No issue found | No hardcoded credentials; workload pods set automountServiceAccountToken=false; metrics endpoint uses authn/authz filters when secure. |
| Config retrieval from ConfigMap | not recorded | No issue found | Reads operator config from a fixed system-namespace ConfigMap; not attacker-controlled. |
| pprof/expvar/statsviz diagnostic server | not recorded | Reported | No additional canonical notes were recorded. |
| CRD spec input validation (WorkspaceName/Models/Patterns) | not recorded | Reported | No additional canonical notes were recorded. |

## Open Questions And Follow Up

- Is the cluster ingress controller exposed to the internet, and does the defaultDomain host other services a workspace could collide with? (Affects severity of the external-exposure findings.)
  - Follow-up prompt: Confirm the ingress controller's external exposure and the configured defaultDomain, then re-rank the Ollama/Open WebUI exposure findings.
- Is the operator's broad RBAC (verbs=\* incl. pods/exec) intended, and is the operator isolated from workload code execution?
  - Follow-up prompt: Review the operator ServiceAccount RBAC and whether workload compromise can reach the operator's in-cluster credentials.
- Are the hack/envoy-sidecar or basic-auth ingress annotations (seen under hack/) deployed in target environments to add auth to the exposed endpoints?
  - Follow-up prompt: Check whether an auth proxy/sidecar is part of the deployed topology in front of the Ollama/Open WebUI ingresses.
