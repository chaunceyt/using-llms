package main

import (
	"context"
	"flag"
	"os/signal"
	"syscall"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	"namespace-watch/internal/config"
	"namespace-watch/internal/controller"
	"namespace-watch/internal/resources"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	restConfig, err := config.Load()
	if err != nil {
		klog.Fatalf("failed to load kubernetes configuration: %v", err)
	}

	kubeClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		klog.Fatalf("failed to create kubernetes client: %v", err)
	}
	dynamicClient, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		klog.Fatalf("failed to create dynamic client: %v", err)
	}

	registry := resources.NewRegistry()
	provisioner := resources.NewProvisioner(dynamicClient, registry)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	informerFactory := informers.NewSharedInformerFactory(kubeClient, 0)
	c := controller.NewController(
		informerFactory.Core().V1().Namespaces(),
		provisioner,
	)

	informerFactory.Start(ctx.Done())

	if err := c.Run(ctx, 2); err != nil {
		klog.Fatalf("controller terminated with error: %v", err)
	}
}
