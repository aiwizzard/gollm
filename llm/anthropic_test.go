package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewAnthropicClient(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		wantErr bool
	}{
		{
			name:    "valid api key",
			apiKey:  "test-key",
			wantErr: false,
		},
		{
			name:    "empty api key",
			apiKey:  "",
			wantErr: false, // Constructor doesn't validate API key
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewAnthropicClient(tt.apiKey)
			if (client == nil) != tt.wantErr {
				t.Errorf("NewAnthropicClient() error = %v, wantErr %v", client == nil, tt.wantErr)
				return
			}
			if !tt.wantErr && client.apiKey != tt.apiKey {
				t.Errorf("NewAnthropicClient() apiKey = %v, want %v", client.apiKey, tt.apiKey)
			}
		})
	}
}

func TestAnthropicClient_Complete(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		statusCode int
		wantErr    bool
		wantResp   *CompletionResponse
	}{
		{
			name: "successful completion",
			response: `{
				"content": [{"type": "text", "text": "Test response"}],
				"model": "claude-3-opus-20240229",
				"stop_reason": "stop"
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantResp: &CompletionResponse{
				Content:      "Test response",
				Model:        "claude-3-opus-20240229",
				FinishReason: "stop",
			},
		},
		{
			name: "API error",
			response: `{
				"error": {
					"message": "Invalid API key"
				}
			}`,
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
			wantResp:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request
				if r.Method != http.MethodPost {
					t.Errorf("Method = %v, want POST", r.Method)
				}
				if r.URL.Path != "/v1/messages" {
					t.Errorf("Path = %v, want /v1/messages", r.URL.Path)
				}
				if r.Header.Get("x-api-key") != "test-key" {
					t.Errorf("x-api-key header = %v, want test-key", r.Header.Get("x-api-key"))
				}
				if r.Header.Get("anthropic-version") != "2023-06-01" {
					t.Errorf("anthropic-version header = %v, want 2023-06-01", r.Header.Get("anthropic-version"))
				}

				// Verify request body
				var reqBody anthropicRequest
				if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
					t.Errorf("Failed to decode request body: %v", err)
				}
				if len(reqBody.Messages) != 1 || reqBody.Messages[0].Role != "user" {
					t.Errorf("Invalid messages in request: %+v", reqBody.Messages)
				}

				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.response))
			}))
			defer server.Close()

			client := &AnthropicClient{
				apiKey:     "test-key",
				httpClient: server.Client(),
			}

			got, err := client.Complete(context.Background(), &CompletionRequest{
				Model:  "claude-3-opus-20240229",
				Prompt: "Test prompt",
			})

			if (err != nil) != tt.wantErr {
				t.Errorf("Complete() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.Content != tt.wantResp.Content {
					t.Errorf("Content = %v, want %v", got.Content, tt.wantResp.Content)
				}
				if got.Model != tt.wantResp.Model {
					t.Errorf("Model = %v, want %v", got.Model, tt.wantResp.Model)
				}
				if got.FinishReason != tt.wantResp.FinishReason {
					t.Errorf("FinishReason = %v, want %v", got.FinishReason, tt.wantResp.FinishReason)
				}
			}
		})
	}
}

func TestAnthropicClient_CompleteStream(t *testing.T) {
	tests := []struct {
		name       string
		responses  []string
		statusCode int
		wantErr    bool
		wantResps  []string
	}{
		{
			name: "successful streaming",
			responses: []string{
				`data: {"content":[{"type":"text","text":"Hello"}],"model":"claude-3-opus-20240229"}` + "\n\n",
				`data: {"content":[{"type":"text","text":" World"}],"model":"claude-3-opus-20240229"}` + "\n\n",
				"data: [DONE]\n\n",
			},
			statusCode: http.StatusOK,
			wantErr:    false,
			wantResps:  []string{"Hello", " World"},
		},
		{
			name:       "API error",
			responses:  []string{},
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
			wantResps:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tt.statusCode != http.StatusOK {
					w.WriteHeader(tt.statusCode)
					return
				}

				flusher, ok := w.(http.Flusher)
				if !ok {
					t.Fatal("Streaming not supported")
				}

				for _, resp := range tt.responses {
					_, err := w.Write([]byte(resp))
					if err != nil {
						t.Fatal(err)
					}
					flusher.Flush()
				}
			}))
			defer server.Close()

			client := &AnthropicClient{
				apiKey:     "test-key",
				httpClient: server.Client(),
			}

			stream, err := client.CompleteStream(context.Background(), &CompletionRequest{
				Model:  "claude-3-opus-20240229",
				Prompt: "Test prompt",
			})

			if (err != nil) != tt.wantErr {
				t.Errorf("CompleteStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			defer stream.Close()

			var responses []string
			for {
				resp, err := stream.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				responses = append(responses, resp.Content)
			}

			if len(responses) != len(tt.wantResps) {
				t.Errorf("Got %d responses, want %d", len(responses), len(tt.wantResps))
			}

			for i, resp := range responses {
				if resp != tt.wantResps[i] {
					t.Errorf("Response %d = %v, want %v", i, resp, tt.wantResps[i])
				}
			}
		})
	}
}
