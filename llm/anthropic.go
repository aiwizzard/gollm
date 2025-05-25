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
)

const (
	anthropicAPIEndpoint = "https://api.anthropic.com/v1/messages"
)

// AnthropicClient implements the LLMProvider interface for Anthropic
type AnthropicClient struct {
	apiKey     string
	httpClient *http.Client
}

// NewAnthropicClient creates a new Anthropic client
func NewAnthropicClient(apiKey string) *AnthropicClient {
	return &AnthropicClient{
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}
}

type anthropicRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float32   `json:"temperature,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content    []contentBlock `json:"content"`
	Model      string         `json:"model"`
	StopReason string         `json:"stop_reason"`
	Error      *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type contentBlock struct {
	Text string `json:"text"`
	Type string `json:"type"`
}

// Complete implements non-streaming completion
func (c *AnthropicClient) Complete(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	anthropicReq := anthropicRequest{
		Model: req.Model,
		Messages: []message{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", anthropicAPIEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var anthropicResp anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, err
	}

	if anthropicResp.Error != nil {
		return nil, fmt.Errorf("anthropic API error: %s", anthropicResp.Error.Message)
	}

	if len(anthropicResp.Content) == 0 {
		return nil, errors.New("no content in response")
	}

	return &CompletionResponse{
		Content:      anthropicResp.Content[0].Text,
		Model:        anthropicResp.Model,
		FinishReason: anthropicResp.StopReason,
	}, nil
}

// anthropicStream implements CompletionStream for Anthropic
type anthropicStream struct {
	reader *bufio.Reader
	closer io.Closer
}

// CompleteStream implements streaming completion
func (c *AnthropicClient) CompleteStream(ctx context.Context, req *CompletionRequest) (CompletionStream, error) {
	anthropicReq := anthropicRequest{
		Model: req.Model,
		Messages: []message{
			{
				Role:    "user",
				Content: req.Prompt,
			},
		},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      true,
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", anthropicAPIEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.apiKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return &anthropicStream{
		reader: bufio.NewReader(resp.Body),
		closer: resp.Body,
	}, nil
}

// Recv implements the CompletionStream interface
func (s *anthropicStream) Recv() (*CompletionResponse, error) {
	line, err := s.reader.ReadBytes('\n')
	if err != nil {
		return nil, err
	}

	if !bytes.HasPrefix(line, []byte("data: ")) {
		return nil, fmt.Errorf("invalid SSE format")
	}

	data := bytes.TrimPrefix(line, []byte("data: "))
	if len(data) == 0 {
		return nil, nil
	}

	var streamResp anthropicResponse
	if err := json.Unmarshal(data, &streamResp); err != nil {
		return nil, err
	}

	if streamResp.Error != nil {
		return nil, fmt.Errorf("anthropic API error: %s", streamResp.Error.Message)
	}

	if len(streamResp.Content) == 0 {
		return nil, nil
	}

	return &CompletionResponse{
		Content:      streamResp.Content[0].Text,
		Model:        streamResp.Model,
		FinishReason: streamResp.StopReason,
	}, nil
}

// Close implements the CompletionStream interface
func (s *anthropicStream) Close() error {
	return s.closer.Close()
}
