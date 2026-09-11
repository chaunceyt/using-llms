// Command namespace-watcher watches Kubernetes namespaces and provisions a set
// of sane default resources (LimitRange, ResourceQuota, a read-only
// ServiceAccount/Role/RoleBinding, and a NetworkPolicy) into every namespace
// that is created.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/example/namespace-watcher/internal/config"
	"github.com/example/namespace-watcher/internal/provisioner"
	"github.com/example/namespace-watcher/internal/watcher"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		kubeconfig  = flag.String("kubeconfig", "", "path to a kubeconfig; if empty, in-cluster config is used, then the default kubeconfig location")
		backfill    = flag.Bool("backfill", false, "also provision namespaces that already exist on startup")
		concurrency = flag.Int("concurrency", 4, "number of workers provisioning namespaces")
		logLevel    = flag.String("log-level", "info", "log level: debug, info, warn or error")
	)
	flag.Parse()

	logger, err := newLogger(*logLevel)
	if err != nil {
		return err
	}
	slog.SetDefault(logger)
	log := logger

	cfg, err := buildRestConfig(*kubeconfig)
	if err != nil {
		return fmt.Errorf("build rest config: %w", err)
	}

	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return fmt.Errorf("build clientset: %w", err)
	}
	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return fmt.Errorf("build dynamic client: %w", err)
	}

	// The watcher creates a fixed, known set of resources, so a static REST
	// mapper is used instead of a runtime discovery call.
	defaults := config.NewDefaultConfig()
	mapper := provisioner.NewStaticMapper(defaults)

	prov := provisioner.New(dynamicClient, mapper, defaults, log)
	w := watcher.New(clientset, prov, watcher.Options{
		Backfill:    *backfill,
		Concurrency: *concurrency,
		Log:         log,
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	log.Info("starting namespace watcher",
		"backfill", *backfill, "concurrency", *concurrency)
	return w.Run(ctx)
}

// newLogger builds a slog.Logger at the requested level, writing to stderr.
func newLogger(level string) (*slog.Logger, error) {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "info":
		lvl = slog.LevelInfo
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		return nil, fmt.Errorf("unknown log level %q (want debug, info, warn or error)", level)
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})), nil
}

// buildRestConfig resolves a rest.Config from (in order): an explicit
// kubeconfig path, the in-cluster service account, or the default kubeconfig
// location (~/.kube/config).
func buildRestConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		return clientcmd.BuildConfigFromFlags("", kubeconfig)
	}
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	cc := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		loadingRules, &clientcmd.ConfigOverrides{})
	return cc.ClientConfig()
}
