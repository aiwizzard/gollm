package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	defaultTimeout = 30 * time.Second
	defaultBaseURL = "https://api.openai.com/v1"
)

// OpenAIConfig contains configuration options for the OpenAI client
type OpenAIConfig struct {
	// APIKey is your OpenAI API key
	APIKey string

	// BaseURL is the base URL for OpenAI API (optional, defaults to https://api.openai.com/v1)
	BaseURL string

	// Timeout is the timeout for API requests (optional, defaults to 30 seconds)
	Timeout time.Duration

	// HTTPClient is a custom HTTP client (optional)
	HTTPClient *http.Client

	// RetryConfig contains retry configuration (optional)
	RetryConfig *RetryConfig
}

// RetryConfig contains configuration for retry behavior
type RetryConfig struct {
	// MaxRetries is the maximum number of retries (default: 3)
	MaxRetries int

	// InitialDelay is the initial delay between retries (default: 1s)
	InitialDelay time.Duration

	// MaxDelay is the maximum delay between retries (default: 5s)
	MaxDelay time.Duration

	// RetryableStatusCodes are the HTTP status codes that should trigger a retry
	RetryableStatusCodes []int
}

// OpenAIClient implements the LLMProvider interface for OpenAI
type OpenAIClient struct {
	config     OpenAIConfig
	httpClient *http.Client
}

// NewOpenAIClient creates a new OpenAI client with the given configuration
func NewOpenAIClient(config OpenAIConfig) *OpenAIClient {
	if config.BaseURL == "" {
		config.BaseURL = defaultBaseURL
	}

	if config.Timeout == 0 {
		config.Timeout = defaultTimeout
	}

	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{
			Timeout: config.Timeout,
		}
	}

	if config.RetryConfig == nil {
		config.RetryConfig = &RetryConfig{
			MaxRetries:   3,
			InitialDelay: time.Second,
			MaxDelay:     5 * time.Second,
			RetryableStatusCodes: []int{
				http.StatusTooManyRequests,
				http.StatusInternalServerError,
				http.StatusBadGateway,
				http.StatusServiceUnavailable,
			},
		}
	}

	return &OpenAIClient{
		config:     config,
		httpClient: config.HTTPClient,
	}
}

// NewOpenAIClientWithKey creates a new OpenAI client with just an API key
func NewOpenAIClientWithKey(apiKey string) *OpenAIClient {
	return NewOpenAIClient(OpenAIConfig{
		APIKey: apiKey,
	})
}

type openaiRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float32         `json:"temperature,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       []Tool          `json:"tools,omitempty"`
	ToolChoice  string          `json:"tool_choice,omitempty"`
}

type openaiMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type openaiResponse struct {
	ID      string   `json:"id"`
	Choices []choice `json:"choices"`
	Model   string   `json:"model"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type choice struct {
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Delta        openaiMessage `json:"delta"`
}

// Complete implements non-streaming completion with retry support
func (c *OpenAIClient) Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	var resp *CompletionResponse
	var lastErr error

	for attempt := 0; attempt <= c.config.RetryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := c.getRetryDelay(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, lastErr = c.complete(ctx, req)
		if lastErr == nil {
			return resp, nil
		}

		if !c.shouldRetry(lastErr) {
			return nil, lastErr
		}
	}

	return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func (c *OpenAIClient) complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	openaiReq := openaiRequest{
		Model: req.Model,
		Messages: []openaiMessage{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stop:        req.Stop,
		Tools:       req.Tools,
	}

	if len(req.Tools) > 0 {
		openaiReq.ToolChoice = "auto"
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/chat/completions", strings.TrimRight(c.config.BaseURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &HTTPError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	var openaiResp openaiResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if openaiResp.Error != nil {
		return nil, fmt.Errorf("OpenAI API error: %s", openaiResp.Error.Message)
	}

	if len(openaiResp.Choices) == 0 {
		return nil, errors.New("no completion choices returned")
	}

	return &CompletionResponse{
		Content:      openaiResp.Choices[0].Message.Content,
		Model:        openaiResp.Model,
		FinishReason: openaiResp.Choices[0].FinishReason,
		ToolCalls:    openaiResp.Choices[0].Message.ToolCalls,
	}, nil
}

func (c *OpenAIClient) shouldRetry(err error) bool {
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) {
		return false
	}

	for _, code := range c.config.RetryConfig.RetryableStatusCodes {
		if httpErr.StatusCode == code {
			return true
		}
	}
	return false
}

func (c *OpenAIClient) getRetryDelay(attempt int) time.Duration {
	delay := c.config.RetryConfig.InitialDelay * time.Duration(1<<uint(attempt-1))
	if delay > c.config.RetryConfig.MaxDelay {
		delay = c.config.RetryConfig.MaxDelay
	}
	return delay
}

// HTTPError represents an HTTP error response
type HTTPError struct {
	StatusCode int
	Message    string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.StatusCode, e.Message)
}

// openAIStream implements CompletionStream for OpenAI
type openAIStream struct {
	reader *bufio.Reader
	closer io.Closer
}

// CompleteStream implements streaming completion
func (c *OpenAIClient) CompleteStream(ctx context.Context, req *CompletionRequest) (CompletionStream, error) {
	openaiReq := openaiRequest{
		Model: req.Model,
		Messages: []openaiMessage{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stop:        req.Stop,
		Stream:      true,
		Tools:       req.Tools,
	}

	if len(req.Tools) > 0 {
		openaiReq.ToolChoice = "auto"
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/chat/completions", strings.TrimRight(c.config.BaseURL, "/"))
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, &HTTPError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	return &openAIStream{
		reader: bufio.NewReader(resp.Body),
		closer: resp.Body,
	}, nil
}

// Recv implements the CompletionStream interface
func (s *openAIStream) Recv() (*CompletionResponse, error) {
	for {
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, fmt.Errorf("failed to read stream: %w", err)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		data := bytes.TrimPrefix(line, []byte("data: "))
		if strings.TrimSpace(string(data)) == "[DONE]" {
			return nil, io.EOF
		}

		var streamResp openaiResponse
		if err := json.Unmarshal(data, &streamResp); err != nil {
			return nil, fmt.Errorf("failed to decode stream response: %w", err)
		}

		if streamResp.Error != nil {
			return nil, fmt.Errorf("OpenAI API error: %s", streamResp.Error.Message)
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		return &CompletionResponse{
			Content:      streamResp.Choices[0].Delta.Content,
			Model:        streamResp.Model,
			FinishReason: streamResp.Choices[0].FinishReason,
			ToolCalls:    streamResp.Choices[0].Delta.ToolCalls,
		}, nil
	}
}

// Close implements the CompletionStream interface
func (s *openAIStream) Close() error {
	return s.closer.Close()
}
