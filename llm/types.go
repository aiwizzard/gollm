package llm

import (
	"context"
	"io"
)

// CompletionRequest represents a request to the LLM
type CompletionRequest struct {
	Prompt      string            `json:"prompt"`
	Model       string            `json:"model"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Temperature float32           `json:"temperature,omitempty"`
	Stop        []string          `json:"stop,omitempty"`
	Options     map[string]string `json:"options,omitempty"`
	Tools       []Tool            `json:"tools,omitempty"`
}

// Tool represents a function that can be called by the model
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents the function definition that can be called by the model
type Function struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

// ToolCall represents a function call made by the model
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// CompletionResponse represents a response from the LLM
type CompletionResponse struct {
	Content      string     `json:"content"`
	Model        string     `json:"model"`
	FinishReason string     `json:"finish_reason,omitempty"`
	ToolCalls    []ToolCall `json:"tool_calls,omitempty"`
}

// LLMProvider interface defines methods that must be implemented by all LLM providers
type LLMProvider interface {
	// Complete makes a non-streaming request to the LLM
	Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompleteStream makes a streaming request to the LLM
	CompleteStream(ctx context.Context, req *CompletionRequest) (CompletionStream, error)
}

// CompletionStream interface for handling streaming responses
type CompletionStream interface {
	// Recv receives the next chunk of the stream
	Recv() (*CompletionResponse, error)

	// Close closes the stream
	io.Closer
}
