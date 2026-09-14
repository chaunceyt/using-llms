# Security Review: AIChatWorkspace Operator (aichat-workspace-operator)

## Scope

The scan reviewed the canonical include paths and exclusions listed below.

- Scan mode: repository
- Target kind: git_revision
- Target ID: aichat-workspace-operator@6bca2ac4706df6e392d748308c1a163d30276b2e
- Revision: 6bca2ac4706df6e392d748308c1a163d30276b2e
- Inventory strategy: repository
- Included paths: none
- Excluded paths: none
- Runtime or test status: not recorded

### Scan Summary

| Field | Value |
| --- | --- |
| Reportable findings | 6 |
| Severity mix | high: 1, medium: 2, low: 3 |
| Confidence mix | high: 4, medium: 2 |
| Coverage | complete |
| Validation mode | not recorded |

Canonical artifacts: `scan-manifest.json`, `findings.json`, and `coverage.json`. This report is a deterministic projection of those files.

## Threat Model

No explicit canonical threat-model summary was recorded.

## Findings

| Finding | Severity | Confidence | Detailed write-up |
| --- | --- | --- | --- |
| [Unvalidated spec.workspaceName enables cross-tenant resource injection and arbitrary namespace deletion via cluster-scoped RBAC](#finding-1) | high | high | inline below |
| [Ollama Modelfile template injection via unescaped spec.models / spec.patterns](#finding-2) | medium | high | inline below |
| [Ollama API and Open WebUI exposed via Ingress without authentication or TLS](#finding-3) | medium | high | inline below |
| [Open WebUI Deployment runs without a hardened SecurityContext (likely as root)](#finding-4) | low | medium | inline below |
| [ConfigMap-driven image tags concatenated into container images without validation](#finding-5) | low | medium | inline below |
| [Cluster-scoped verbs=\* on pods/pods/exec and shell exec sink (least-privilege / latent command injection)](#finding-6) | low | high | inline below |

### Confidence Scale

| Label | Meaning |
| --- | --- |
| high | Direct evidence supports the finding with no material unresolved blocker. |
| medium | Evidence supports a plausible issue, but material runtime or reachability proof remains. |
| low | Evidence is incomplete and the item is retained only for explicit follow-up. |

<a id="finding-1"></a>

### [1] Unvalidated spec.workspaceName enables cross-tenant resource injection and arbitrary namespace deletion via cluster-scoped RBAC

| Field | Value |
| --- | --- |
| Severity | high |
| Confidence | high |
| Confidence rationale | Source-confirmed dataflow: workspaceName flows unchanged from api types into every ensure\* New\* constructor as the target namespace (handle_reconcile.go), adoption of existing namespaces is verified in namespace.go, and unconditional Namespace DELETE on teardown is in aichatworkspace_controller.go; operator ClusterRole grants verbs=\* on namespaces. |
| Category | Broken Access Control / Tenant Isolation |
| CWE | CWE-284 |
| Affected lines | internal/controller/handle_reconcile.go:47-131, internal/controller/namespace.go:36-64, internal/controller/aichatworkspace_controller.go:232-248, api/v1alpha1/aichatworkspace_types.go:28-31, config/rbac/role.yaml:14-24 |

#### Summary

The controller uses the user-controlled AIChatWorkspace.spec.workspaceName verbatim as the Kubernetes namespace in which it creates namespaces, ResourceQuotas, PVCs, ServiceAccounts, StatefulSets, Deployments, Services and Ingresses, and on CR deletion issues a Namespace DELETE for that exact name. ensureNamespace adopts an existing namespace silently (it only creates when NotFound), so a CR whose workspaceName matches any pre-existing namespace causes the operator's privileged service account to inject workloads and quotas into that namespace; deleting such a CR deletes the entire namespace and everything in it.

#### Root Cause

No validation ties spec.workspaceName to the CR's own namespace or owner, and no ownership check is made before the controller uses it as a target namespace (creation) or deletes it (teardown), while the operator holds cluster-scoped verbs=\*.

**Code evidence 1** — `api/v1alpha1/aichatworkspace_types.go:28-31`

```
WorkspaceName string `json:"workspaceName"` // +kubebuilder:validation:XValidation:rule="self == oldSelf"
```

**Code evidence 2** — `internal/controller/namespace.go:36-64`

```
func (r *AIChatWorkspaceReconciler) ensureNamespace(...){
  err := r.Get(context.TODO(), types.NamespacedName{Name: ns.Name}, found)
  if err != nil && errors.IsNotFound(err) { ... create ... } else if err != nil { ... }
  return nil, nil // existing namespace adopted; reconcile proceeds
}
```

#### Validation

Confirmed by direct source review: handle_reconcile.go passes aichat.Spec.WorkspaceName as Namespace to NewNamespace/NewResourceQuota/NewPersistentVolumeClaim/NewServiceAccount/NewStatefulSet/NewDeployment/NewIngress; namespace.go only creates when NotFound, so pre-existing namespaces are adopted; deleteAIChatWorkspace deletes the raw Namespace. The CEL rule is immutability-only and does not constrain allowed values; no validating admission webhook is registered.

**Code evidence 1** — `internal/controller/namespace.go:36-64`

```
func (r *AIChatWorkspaceReconciler) ensureNamespace(...){
  err := r.Get(context.TODO(), types.NamespacedName{Name: ns.Name}, found)
  if err != nil && errors.IsNotFound(err) { ... create ... } else if err != nil { ... }
  return nil, nil // existing namespace adopted; reconcile proceeds
}
```

**Code evidence 2** — `internal/controller/aichatworkspace_controller.go:232-248`

```
namespace := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: instance.Spec.WorkspaceName}}
err := r.Delete(context.TODO(), namespace)
```

#### Dataflow

CR spec.workspaceName -\> reconcile -\> ensure\*/New\* (Namespace target) or deleteAIChatWorkspace (Namespace DELETE) -\> kubernetes API via privileged manager SA.

#### Reachability

Attacker-controlled CR field reaches cluster-scoped API actions with no intermediate control; requires only create+delete on the aichatworkspaces resource.

#### Severity

**High** — The scan assigned high severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Tie workspaceName to ownership: require it to equal the CR's own metadata.namespace (CEL rule such as `self == object.metadata.namespace`) or enforce a strict RFC1123 pattern + allowlist via a validating admission webhook; reject names of pre-existing, non-operator-owned namespaces; scope the manager ClusterRole away from verbs=\* on namespaces and drop namespace 'delete'; on teardown delete only resources the controller created (ownerReferences/labels) instead of DELETE-ing the raw Namespace.

<a id="finding-2"></a>

### [2] Ollama Modelfile template injection via unescaped spec.models / spec.patterns

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | Direct source confirmation: raw fmt.Sprintf of untrusted strings into a Modelfile template with no escaping; CRD schema imposes no constraints on models/patterns. |
| Category | Injection (unsafe code generation) |
| CWE | CWE-74 |
| Affected lines | internal/adapters/ai/modelfiles/modelfiles.go:38-52, internal/adapters/ollama/ollama.go:287-319, internal/controller/ollama.go:106-131 |

#### Summary

prompt() in modelfiles.go builds an Ollama Modelfile with fmt.Sprintf embedding the user-supplied model into `FROM %s` and pattern into `SYSTEM """%s"""` with no escaping or sanitization. The CRD imposes no pattern/maxLength on models/patterns, so newlines and triple-quotes can break out and inject arbitrary Modelfile directives (e.g. a different FROM base model or extra TEMPLATE/PARAMETER lines), which the controller sends to /api/create.

#### Root Cause

User-controlled model/pattern strings are interpolated verbatim into an Ollama Modelfile without neutralization, allowing directive injection.

**Code evidence 1** — `internal/adapters/ai/modelfiles/modelfiles.go:38-52`

```
var promptTemplate = `
FROM %s
...
SYSTEM """
%s
"""`
return fmt.Sprintf(promptTemplate, model, pattern)
```

#### Validation

No escaping or sanitization exists; a pattern containing a newline plus `"""` terminates the SYSTEM block and injects directives. Impact is confined to the tenant's own per-workspace Ollama StatefulSet (no cross-tenant path), but allows attacker-chosen base models / in-cluster fetch targets via FROM.

**Code evidence 1** — `internal/adapters/ai/modelfiles/modelfiles.go:38-52`

```
var promptTemplate = `
FROM %s
...
SYSTEM """
%s
"""`
return fmt.Sprintf(promptTemplate, model, pattern)
```

#### Dataflow

CR spec.models/spec.patterns -\> ensureStatefulSet -\> ollama.CreateFromModelFile -\> modelfiles.GetSystemPromptPattern -\> Ollama /api/create.

#### Reachability

Trivially reachable from CR spec with no intermediate control; bounded by per-workspace namespace/ResourceQuota isolation.

#### Severity

**Medium** — The scan assigned medium severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Validate/sanitize spec.models and spec.patterns at admission time (CRD CEL pattern disallowing newlines and `"""`, maxLength) and additionally escape/neutralize triple-quotes, backslashes and newlines in prompt(); prefer building the Modelfile from structured fields rather than string interpolation; never concatenate untrusted text into directives such as FROM.

<a id="finding-3"></a>

### [3] Ollama API and Open WebUI exposed via Ingress without authentication or TLS

| Field | Value |
| --- | --- |
| Severity | medium |
| Confidence | high |
| Confidence rationale | NewIngress clearly omits tls and auth; setIngressDNSHost builds predictable public hosts; README documents auth as unimplemented. External reachability depends on cluster ingress exposure, noted as a deployment prerequisite. |
| Category | Missing Authentication / Security Misconfiguration |
| CWE | CWE-306 |
| Affected lines | internal/adapters/k8s/common.go:189-230, internal/controller/handle_reconcile.go:121-134, internal/controller/handle_reconcile.go:153-163 |

#### Summary

handle_reconcile.go creates Ingresses for both workloads at \<workspace\>.defaultDomain (openwebui) and \<workspace\>-api.defaultDomain (ollama). NewIngress defines a path '/' rule with no spec.tls block and no auth annotations, routing to the unauthenticated Ollama API (port 11434) and Open WebUI. The README lists adding auth to the Ollama endpoint as unimplemented roadmap, confirming the surface is intentionally open.

#### Root Cause

The generated Ingress omits both transport encryption and authentication while exposing the unauthenticated Ollama management/inference API.

**Code evidence 1** — `internal/adapters/k8s/common.go:189-230`

```
return &networkingv1.Ingress{ Spec: networkingv1.IngressSpec{ Rules: []networkingv1.IngressRule{{ Host: hostname, ... }} } } // no TLS block, no auth annotations
```

#### Validation

No tls or auth field is set in NewIngress; README.md:51 marks 'Add auth to the Ollama endpoint' as unimplemented. The KEDA interceptor-proxy path that could front it is commented out (handle_reconcile.go:136-140).

**Code evidence 1** — `internal/adapters/k8s/common.go:189-230`

```
return &networkingv1.Ingress{ Spec: networkingv1.IngressSpec{ Rules: []networkingv1.IngressRule{{ Host: hostname, ... }} } } // no TLS block, no auth annotations
```

#### Dataflow

Internet -\> Ingress host / \<workspace\>-api.\<domain\> -\> Ollama Service:11434 (unauthenticated) or Open WebUI :8080.

#### Reachability

Reachability depends on an externally routed ingress controller for the cluster's defaultDomain; source establishes the exposed surface.

#### Severity

**Medium** — The scan assigned medium severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Add spec.tls with a managed certificate per ingress host; front the ingresses with an authentication layer (OIDC/basic-auth sidecar or ingress-nginx auth annotation); require bearer-token auth on the Ollama API or bind it to localhost and expose only through an authenticated proxy; restrict /api/pull, /api/create, /api/delete to operator service accounts; add NetworkPolicy limiting traffic to the ingress controller; disable open signup in Open WebUI.

<a id="finding-4"></a>

### [4] Open WebUI Deployment runs without a hardened SecurityContext (likely as root)

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | SecurityContext is demonstrably omitted (commented out); whether the container actually executes as UID 0 depends on the upstream image's default USER directive. |
| Category | Execution with Unnecessary Privileges |
| CWE | CWE-250 |
| Affected lines | internal/adapters/k8s/apps.go:66-75, internal/adapters/k8s/apps.go:188-199 |

#### Summary

NewDeployment for Open WebUI comments out both the pod and container SecurityContext, leaving the internet-exposed web UI to run with the upstream image's default user/capabilities. In contrast the Ollama StatefulSet applies defaultSecurityContext()/defaultPodSecurityContext() (RunAsNonRoot, drop ALL caps, readonly rootfs, RuntimeDefault seccomp).

#### Root Cause

The internet-facing Open WebUI deployment omits the container/pod security context used on Ollama, running with unnecessary privileges and capabilities.

**Code evidence 1** — `internal/adapters/k8s/apps.go:66-75`

```
// at the moment having issues getting open webui to run as non-root
// SecurityContext: defaultPodSecurityContext(),
...
// SecurityContext: defaultSecurityContext(),
```

#### Validation

apps.go:70 and :75 comment out the hardened contexts; automountServiceAccountToken=false is set (mitigating token exposure). Defense-in-depth gap that amplifies the ingress-exposure finding.

**Code evidence 1** — `internal/adapters/k8s/apps.go:66-75`

```
// at the moment having issues getting open webui to run as non-root
// SecurityContext: defaultPodSecurityContext(),
...
// SecurityContext: defaultSecurityContext(),
```

#### Dataflow

Compromise of open-webui container -\> runs as default (root-capable) user with default capabilities.

#### Reachability

Depends on a code-execution primitive in Open WebUI or its image; lowers the cost/impact of such a compromise.

#### Severity

**Low** — The scan assigned low severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Re-enable defaultSecurityContext()/defaultPodSecurityContext() on the open-webui Deployment with a fixed runAsUser/runAsNonRoot consistent with the image, drop all capabilities, and apply RuntimeDefault seccomp.

<a id="finding-5"></a>

### [5] ConfigMap-driven image tags concatenated into container images without validation

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | medium |
| Confidence rationale | Raw configmap strings are interpolated into image references (confirmed), but exploitation requires write access to the operator's cluster-scoped ConfigMap, which is normally admin-only. |
| Category | Supply Chain / Configuration |
| CWE | CWE-1357 |
| Affected lines | internal/config/config.go:44-71, internal/adapters/k8s/apps.go:49-154 |

#### Summary

config.GetConfig reads openwebUIImageTag/ollamaImageTag from the operator ConfigMap with no syntax or registry-allowlist validation and concatenates them into `repo:tag` image references applied to every workspace's StatefulSet/Deployment. A malformed tag could alter which registry/image is pulled.

#### Root Cause

Container image tags and DNS hostnames are read from config without format/registry validation before being applied cluster-wide.

**Code evidence 1** — `internal/adapters/k8s/apps.go:49-154`

```
containerImage := fmt.Sprintf("%s:%s", constants.OpenwebuiContainerImageName, openwebuiContainerImageTag)
```

#### Validation

getConfigMapString returns the raw value with no check; NewStatefulSet/NewDeployment interpolate it directly. Image names are fixed to upstream constants, limiting but not removing substitution of an attacker-controlled tag.

**Code evidence 1** — `internal/adapters/k8s/apps.go:49-154`

```
containerImage := fmt.Sprintf("%s:%s", constants.OpenwebuiContainerImageName, openwebuiContainerImageTag)
```

#### Dataflow

ConfigMap openwebUIImageTag/ollamaImageTag -\> GetConfig -\> NewDeployment/NewStatefulSet image string.

#### Reachability

Requires admin-level write access to the operator namespace's ConfigMap; not reachable by tenants via the CR.

#### Severity

**Low** — The scan assigned low severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Pin images by digest instead of mutable tags, validate tag values in GetConfig against a strict allowlist regex (e.g. ^\[A-Za-z0-9_.-\]+$), and restrict write access to the operator ConfigMap; validate defaultDomain against an allowed suffix.

<a id="finding-6"></a>

### [6] Cluster-scoped verbs=\* on pods/pods/exec and shell exec sink (least-privilege / latent command injection)

| Field | Value |
| --- | --- |
| Severity | low |
| Confidence | high |
| Confidence rationale | Verified the exec command is constants-derived today (no attacker-controlled input reaches it); the verbs=\* grant and shell invocation are confirmed in source. |
| Category | Improper Privilege Management / least privilege |
| CWE | CWE-269 |
| Affected lines | internal/adapters/k8s/common.go:304-343, config/rbac/role.yaml:14-24 |

#### Summary

The manager ClusterRole grants verbs=\* on pods and pods/exec across all namespaces, and ExecuteRemoteCommand runs '/bin/sh -c \<command\>' inside tenant pods via the exec subresource. Today the command is built only from constant mount paths (du -sh), so it is not attacker-controlled; the risk is over-broad privilege plus a latent injection sink for any future caller that concatenates tainted data.

#### Root Cause

Operator holds cluster-wide pods/exec verbs=\* and routes status collection through a full shell, exceeding the minimal privilege needed for the read-only du operation.

**Code evidence 1** — `internal/adapters/k8s/common.go:304-343`

```
VersionedParams(&corev1.PodExecOptions{ Command: []string{"/bin/sh", "-c", command}, ... })
```

#### Validation

The only callers pass constant mount paths (duSHCommand(constants.OllamaVolumeMountPath / OpenwebuiVolumeMountPath)); no CR field or workspace name reaches the exec command, so there is no current injection.

**Code evidence 1** — `internal/adapters/k8s/common.go:304-343`

```
VersionedParams(&corev1.PodExecOptions{ Command: []string{"/bin/sh", "-c", command}, ... })
```

#### Dataflow

getPodInfo -\> duSHCommand(constant) -\> ExecuteRemoteCommand '/bin/sh -c du -sh \<mount\>'.

#### Reachability

Latent; requires a future caller passing untrusted data or an operator-pod compromise.

#### Severity

**Low** — The scan assigned low severity; no separate canonical severity rationale was recorded.

Additional runtime or deployment evidence could raise or lower this severity.

#### Remediation

Tighten RBAC to minimal verbs and scope pods/exec to operator-owned namespaces (or drop exec in favor of metrics/filesystem APIs); invoke the specific binary without a full shell where feasible; never concatenate tainted data into the exec command.

## Reviewed Surfaces

| Surface | Risk Area | Outcome | Notes |
| --- | --- | --- | --- |
| internal/controller - AIChatWorkspace reconciliation, resource lifecycle, namespace/ingress/ollama/openwebui handling | not recorded | Reported | No additional canonical notes were recorded. |
| internal/adapters/ollama - model pull/create/list via Ollama SDK | not recorded | Reported | No additional canonical notes were recorded. |
| internal/adapters/k8s - resource constructors (Namespace, ServiceAccount, PVC, Service, Ingress, StatefulSet, Deployment) and pod exec helper | not recorded | Reported | No additional canonical notes were recorded. |
| internal/adapters/ai/modelfiles - Ollama Modelfile generation | not recorded | Reported | No additional canonical notes were recorded. |
| internal/config and internal/constants - operator configuration from ConfigMap, image tags, domain | not recorded | Reported | No additional canonical notes were recorded. |
| api/v1alpha1 - AIChatWorkspace CRD types and CEL validation rules | not recorded | Reported | No additional canonical notes were recorded. |
| cmd/main.go - manager setup, metrics/probe servers, webhook server, pprof (localhost-bound) | not recorded | No issue found | No additional canonical notes were recorded. |
| config/rbac - operator manager ClusterRole and editor/viewer roles | not recorded | Reported | No additional canonical notes were recorded. |

## Open Questions And Follow Up

- Whether AIChatWorkspace CR creation is actually granted to untrusted tenants in production (vs. platform-engineers only) determines the practical likelihood of the workspaceName tenant-isolation finding; no RBAC binding granting the editor role to end users is committed.
- Whether the cluster ingress controller routes the generated \<workspace\>\[-api\].defaultDomain hosts publicly determines the real-world exposure of the unauthenticated Ollama API; this depends on cluster DNS/ingress configuration outside the repository.
