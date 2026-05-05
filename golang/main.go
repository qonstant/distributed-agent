package main

import (
	"log"
	"os"
	"strings"

	app "github.com/qonstant/distributed-agent/internal/app"
)

func main() {
	mode := strings.TrimSpace(os.Getenv("APP_MODE"))
	if mode == "" {
		mode = "bot"
	}

	var err error
	switch mode {
	case "bot":
		err = app.Run()
	case "events-worker":
		err = app.RunTurnWorker()
	default:
		log.Fatalf("unknown APP_MODE %q", mode)
	}

	if err != nil {
		log.Fatal(err)
	}
}
