package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"k8s.io/client-go/dynamic"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	"namespace-watch-dv4f/internal/controller"
)

func main() {
	klog.InitFlags(nil)

	kubeconfig := flag.String("kubeconfig", "",
		"path to a kubeconfig; uses in-cluster config when empty, then $KUBECONFIG/~/.kube/config")
	resyncPeriod := flag.Duration("resync-period", 0,
		"how often to re-list namespaces and re-ensure defaults; 0 disables resync")
	workers := flag.Int("workers", 2, "number of sync workers")
	skipPrefixes := flag.String("skip-prefixes", "kube-",
		"comma-separated namespace name prefixes to skip")
	flag.Parse()

	cfg, err := buildRESTConfig(*kubeconfig)
	if err != nil {
		fmt.Fprintf(os.Stderr, "building kube config: %v\n", err)
		os.Exit(1)
	}

	// Client-go defaults to very low rate limits; a namespace burst would throttle hard.
	cfg.QPS = 50
	cfg.Burst = 100

	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "building dynamic client: %v\n", err)
		os.Exit(1)
	}

	informerFactory := kubeinformers.NewSharedInformerFactory(
		kubernetes.NewForConfigOrDie(cfg), *resyncPeriod)

	ctrl := controller.New(dyn, informerFactory,
		strings.Split(*skipPrefixes, ","))

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if err := ctrl.Run(ctx, *workers); err != nil {
		fmt.Fprintf(os.Stderr, "controller stopped: %v\n", err)
		os.Exit(1)
	}
}

func buildRESTConfig(kubeconfigPath string) (*rest.Config, error) {
	if kubeconfigPath != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	}
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
}
