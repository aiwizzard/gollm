package chat

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/aiwizzard/gollm/llm"
)

// RunExample demonstrates basic chat completions without tool calls
func RunExample() error {
	// Get API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	// Initialize OpenAI client with custom configuration
	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  apiKey,
		Timeout: 30 * time.Second,
		RetryConfig: &llm.RetryConfig{
			MaxRetries:   3,
			InitialDelay: time.Second,
			MaxDelay:     5 * time.Second,
		},
	})

	// Example 1: Basic non-streaming completion
	if err := runBasicCompletion(client); err != nil {
		return fmt.Errorf("basic completion failed: %w", err)
	}

	// Example 2: Streaming completion
	if err := runStreamingCompletion(client); err != nil {
		return fmt.Errorf("streaming completion failed: %w", err)
	}

	// Example 3: Completion with temperature and max tokens
	if err := runCustomizedCompletion(client); err != nil {
		return fmt.Errorf("customized completion failed: %w", err)
	}

	return nil
}

func runBasicCompletion(client *llm.OpenAIClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("\nBasic Completion Example:")
	resp, err := client.Complete(ctx, &llm.CompletionRequest{
		Model:  "gpt-4",
		Prompt: "Explain quantum computing in one sentence.",
	})
	if err != nil {
		return fmt.Errorf("completion request failed: %w", err)
	}

	fmt.Printf("Response: %s\n", resp.Content)
	return nil
}

func runStreamingCompletion(client *llm.OpenAIClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("\nStreaming Completion Example:")
	stream, err := client.CompleteStream(ctx, &llm.CompletionRequest{
		Model:  "gpt-4",
		Prompt: "Write a haiku about programming.",
	})
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	defer stream.Close()

	fmt.Print("Response: ")
	for {
		chunk, err := stream.Recv()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return fmt.Errorf("error receiving stream: %w", err)
		}

		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}
	fmt.Println()
	return nil
}

func runCustomizedCompletion(client *llm.OpenAIClient) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("\nCustomized Completion Example:")
	resp, err := client.Complete(ctx, &llm.CompletionRequest{
		Model:       "gpt-4",
		Prompt:      "Generate a creative name for a tech startup.",
		MaxTokens:   20,            // Limit response length
		Temperature: 0.8,           // Higher temperature for more creative responses
		Stop:        []string{"."}, // Stop at the first period
	})
	if err != nil {
		return fmt.Errorf("completion request failed: %w", err)
	}

	fmt.Printf("Response: %s\n", resp.Content)
	return nil
}
