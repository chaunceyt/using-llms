// Package config centralizes the "sane defaults" that the namespace watcher
// applies to every namespace it provisions. All the values that are expected
// to change over time live here so they can be tweaked in one place (or
// overridden programmatically) without touching the provisioning logic.
package config

// DefaultLabelKey is applied to every resource the watcher creates so the
// resources it owns are easy to discover and reconcile later.
const DefaultLabelKey = "app.kubernetes.io/managed-by"

// DefaultLabelValue is the value paired with DefaultLabelKey.
const DefaultLabelValue = "namespace-watcher"

// Defaults describes the full set of resources created for a new namespace
// along with their default values.
type Defaults struct {
	// Labels are stamped onto the metadata of every resource that is created.
	// The managed-by label is always present; extra labels can be added here.
	Labels map[string]string

	LimitRange    LimitRangeDefaults
	ResourceQuota ResourceQuotaDefaults
	RBAC          RBACDefaults
	NetworkPolicy NetworkPolicyDefaults
}

// LimitRangeDefaults holds the per-container CPU/memory defaults.
type LimitRangeDefaults struct {
	Name                 string
	DefaultMemory        string
	DefaultCPU           string
	DefaultRequestMemory string
	DefaultRequestCPU    string
	MaxMemory            string
	MaxCPU               string
}

// ResourceQuotaDefaults holds the aggregate namespace quota.
type ResourceQuotaDefaults struct {
	Name                   string
	RequestsCPU            string
	RequestsMemory         string
	LimitsCPU              string
	LimitsMemory           string
	Pods                   int64
	Services               int64
	PersistentVolumeClaims int64
}

// RBACDefaults holds the ServiceAccount/Role/RoleBinding names and the rules
// the Role grants (read-only access to configmaps by default).
type RBACDefaults struct {
	ServiceAccountName string
	RoleName           string
	RoleBindingName    string

	APIGroups []string
	Resources []string
	Verbs     []string
}

// NetworkPolicyDefaults holds the NetworkPolicy shape: which pod it applies to
// and the ingress/egress rules it enforces.
type NetworkPolicyDefaults struct {
	Name        string
	PodSelector map[string]string
	PolicyTypes []string

	Ingress IngressRule
	Egress  EgressRule

	// AllowDNS adds an egress rule permitting UDP/53 to any namespace.
	AllowDNS bool
	DNSPort  int32
}

// IngressRule describes the single allowed ingress rule.
type IngressRule struct {
	// FromApp is the value of the "app" label on the source pods that may
	// reach this namespace (e.g. an ingress controller).
	FromApp  string
	Port     int32
	Protocol string
}

// EgressRule describes the single allowed egress rule (besides optional DNS).
type EgressRule struct {
	// ToApp is the value of the "app" label on the destination pods this
	// namespace may reach (e.g. a database).
	ToApp    string
	Port     int32
	Protocol string
}

// NewDefaultConfig returns the defaults exactly as specified in the
// requirements. Namespace-scoped resource names are fixed; the namespace
// itself is supplied at provisioning time.
func NewDefaultConfig() Defaults {
	labels := map[string]string{
		DefaultLabelKey: DefaultLabelValue,
	}

	return Defaults{
		Labels: labels,

		LimitRange: LimitRangeDefaults{
			Name:                 "default-limits",
			DefaultMemory:        "256Mi",
			DefaultCPU:           "500m",
			DefaultRequestMemory: "128Mi",
			DefaultRequestCPU:    "100m",
			MaxMemory:            "2Gi",
			MaxCPU:               "2",
		},

		ResourceQuota: ResourceQuotaDefaults{
			Name:                   "production-quota",
			RequestsCPU:            "20",
			RequestsMemory:         "40Gi",
			LimitsCPU:              "40",
			LimitsMemory:           "80Gi",
			Pods:                   100,
			Services:               20,
			PersistentVolumeClaims: 30,
		},

		RBAC: RBACDefaults{
			ServiceAccountName: "myapp-sa",
			RoleName:           "myapp-role",
			RoleBindingName:    "myapp-rolebinding",
			APIGroups:          []string{""},
			Resources:          []string{"configmaps"},
			Verbs:              []string{"get", "list"},
		},

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
	}
}
