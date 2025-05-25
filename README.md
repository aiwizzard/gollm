# GoLLM

A Go library for interacting with various LLM providers (OpenAI and Anthropic) with support for both streaming and non-streaming responses.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-3.5, GPT-4)
  - Anthropic (Claude)
- Streaming and non-streaming responses
- Simple, unified interface
- Type-safe responses
- Error handling

## Installation

```bash
go get github.com/aiwizzard/gollm
```

## Usage

```go
package main

import (
    "context"
    "fmt"
    "github.com/aiwizzard/gollm/llm"
)

func main() {
    // Initialize OpenAI client
    openaiClient := llm.NewOpenAIClient("your-api-key")
    
    // Make a non-streaming request
    resp, err := openaiClient.Complete(context.Background(), &llm.CompletionRequest{
        Prompt: "Tell me a joke",
        Model:  "gpt-3.5-turbo",
    })
    
    // Make a streaming request
    stream, err := openaiClient.CompleteStream(context.Background(), &llm.CompletionRequest{
        Prompt: "Tell me a story",
        Model:  "gpt-3.5-turbo",
    })
    
    for {
        chunk, err := stream.Recv()
        if err != nil {
            break
        }
        fmt.Print(chunk.Content)
    }
}
```
