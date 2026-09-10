// Package config resolves the Kubernetes client configuration used by the
// watcher, preferring an explicit kubeconfig and falling back to in-cluster.
package config

import (
	"flag"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"
)

var kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file (in-cluster config is used when empty)")

// Load returns a rest.Config for talking to the cluster. It honours an
// explicit --kubeconfig flag, otherwise uses in-cluster credentials.
func Load() (*rest.Config, error) {
	if *kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", *kubeconfig)
	}

	if cfg, err := rest.InClusterConfig(); err == nil {
		klog.Info("using in-cluster configuration")
		return cfg, nil
	}

	// Fall back to the default kubeconfig search path (e.g. ~/.kube/config).
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
}
