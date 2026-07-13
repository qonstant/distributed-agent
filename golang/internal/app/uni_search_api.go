package app

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/qonstant/distributed-agent/internal/adapter/postgres"
	"github.com/qonstant/distributed-agent/internal/adapter/unisearchhttp"
	"github.com/qonstant/distributed-agent/internal/config"
)

func RunUniSearchAPI() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	if err := cfg.ValidateUniSearchAPI(); err != nil {
		return err
	}

	repo, err := postgres.NewUniversitySearchRepository(cfg.UniDatabaseURL)
	if err != nil {
		return fmt.Errorf("failed to initialize university search repository: %w", err)
	}
	defer repo.Close()

	server := &http.Server{
		Addr:              cfg.UniSearchListenAddr,
		Handler:           unisearchhttp.NewServer(repo).Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Printf("Uni search API listening on %s", cfg.UniSearchListenAddr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("uni search api server failed: %w", err)
		}
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown uni search api: %w", err)
	}

	log.Println("Uni search API stopped")
	return nil
}
