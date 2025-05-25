package main

import (
	"log"
	"os"

	"github.com/aiwizzard/gollm/examples/chat"
	"github.com/aiwizzard/gollm/examples/tools"
)

func main() {
	// Run the tools example
	log.Println("Running tools example...")
	if err := tools.RunExample(); err != nil {
		log.Printf("Error running tools example: %v", err)
		os.Exit(1)
	}

	// Run the basic chat example
	log.Println("\nRunning basic chat example...")
	if err := chat.RunExample(); err != nil {
		log.Printf("Error running chat example: %v", err)
		os.Exit(1)
	}
}
