package main

import (
	"log"

	app "github.com/qonstant/distributed-agent/internal/app"
)

func main() {
	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}
