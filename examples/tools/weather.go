package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/aiwizzard/gollm/llm"
)

// WeatherParams represents the parameters for the getWeather function
type WeatherParams struct {
	Location string `json:"location"`
	Unit     string `json:"unit"`
}

// GetWeather simulates getting weather data
func GetWeather(location, unit string) string {
	// This is a mock implementation
	return fmt.Sprintf("The weather in %s is 22°%s", location, unit)
}

// RunExample demonstrates how to use tools with the LLM
func RunExample() error {
	// Get API key from environment variable
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}

	// Initialize OpenAI client with custom configuration
	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  apiKey,
		Timeout: 60 * time.Second,
		RetryConfig: &llm.RetryConfig{
			MaxRetries:   5,
			InitialDelay: 2 * time.Second,
			MaxDelay:     10 * time.Second,
		},
	})

	// Define the tool (function) that the model can use
	weatherTool := llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        "get_weather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state, e.g., San Francisco, CA",
					},
					"unit": map[string]interface{}{
						"type": "string",
						"enum": []string{"C", "F"},
					},
				},
				"required": []string{"location", "unit"},
			},
		},
	}

	if err := runNonStreamingExample(client, weatherTool); err != nil {
		return fmt.Errorf("non-streaming example failed: %w", err)
	}

	if err := runStreamingExample(client, weatherTool); err != nil {
		return fmt.Errorf("streaming example failed: %w", err)
	}

	return nil
}

func runNonStreamingExample(client *llm.OpenAIClient, weatherTool llm.Tool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Make a request that will trigger the tool
	resp, err := client.Complete(ctx, &llm.CompletionRequest{
		Model:  "gpt-4",
		Prompt: "What's the weather like in London? Please use Celsius.",
		Tools:  []llm.Tool{weatherTool},
	})
	if err != nil {
		return fmt.Errorf("completion request failed: %w", err)
	}

	// Handle tool calls if any
	if len(resp.ToolCalls) > 0 {
		// Process each tool call
		for _, call := range resp.ToolCalls {
			if call.Function.Name == "get_weather" {
				// Parse the arguments
				var params WeatherParams
				if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
					return fmt.Errorf("failed to parse weather parameters: %w", err)
				}

				// Call the actual function
				result := GetWeather(params.Location, params.Unit)

				// Send the result back to the model
				resp, err = client.Complete(ctx, &llm.CompletionRequest{
					Model: "gpt-4",
					Prompt: fmt.Sprintf("The weather function returned: %s. "+
						"Please provide a natural response to the user's question.", result),
				})
				if err != nil {
					return fmt.Errorf("follow-up completion request failed: %w", err)
				}
			}
		}
	}

	fmt.Printf("Response: %s\n", resp.Content)
	return nil
}

func runStreamingExample(client *llm.OpenAIClient, weatherTool llm.Tool) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("\nStreaming example:")
	stream, err := client.CompleteStream(ctx, &llm.CompletionRequest{
		Model:  "gpt-4",
		Prompt: "What's the weather like in Paris? Please use Fahrenheit.",
		Tools:  []llm.Tool{weatherTool},
	})
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	defer stream.Close()

	// Process the streaming response
	var toolCalls []llm.ToolCall
	for {
		chunk, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error receiving stream: %w", err)
		}

		// Accumulate tool calls
		if len(chunk.ToolCalls) > 0 {
			toolCalls = append(toolCalls, chunk.ToolCalls...)
		}

		// Print content if any
		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}

	// Handle tool calls from streaming response
	if len(toolCalls) > 0 {
		for _, call := range toolCalls {
			if call.Function.Name == "get_weather" {
				var params WeatherParams
				if err := json.Unmarshal([]byte(call.Function.Arguments), &params); err != nil {
					return fmt.Errorf("failed to parse weather parameters: %w", err)
				}

				result := GetWeather(params.Location, params.Unit)

				stream, err = client.CompleteStream(ctx, &llm.CompletionRequest{
					Model: "gpt-4",
					Prompt: fmt.Sprintf("The weather function returned: %s. "+
						"Please provide a natural response to the user's question.", result),
				})
				if err != nil {
					return fmt.Errorf("follow-up streaming request failed: %w", err)
				}
				defer stream.Close()

				for {
					chunk, err := stream.Recv()
					if err != nil {
						if err == io.EOF {
							break
						}
						return fmt.Errorf("error receiving stream: %w", err)
					}
					fmt.Print(chunk.Content)
				}
			}
		}
	}

	return nil
}
