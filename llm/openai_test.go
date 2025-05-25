package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewOpenAIClient(t *testing.T) {
	tests := []struct {
		name   string
		config OpenAIConfig
		want   *OpenAIClient
	}{
		{
			name: "default configuration",
			config: OpenAIConfig{
				APIKey: "test-key",
			},
			want: &OpenAIClient{
				config: OpenAIConfig{
					APIKey:  "test-key",
					BaseURL: defaultBaseURL,
					Timeout: defaultTimeout,
					RetryConfig: &RetryConfig{
						MaxRetries:   3,
						InitialDelay: time.Second,
						MaxDelay:     5 * time.Second,
					},
				},
			},
		},
		{
			name: "custom configuration",
			config: OpenAIConfig{
				APIKey:  "test-key",
				BaseURL: "https://custom.api.com",
				Timeout: 60 * time.Second,
				RetryConfig: &RetryConfig{
					MaxRetries:   5,
					InitialDelay: 2 * time.Second,
					MaxDelay:     10 * time.Second,
				},
			},
			want: &OpenAIClient{
				config: OpenAIConfig{
					APIKey:  "test-key",
					BaseURL: "https://custom.api.com",
					Timeout: 60 * time.Second,
					RetryConfig: &RetryConfig{
						MaxRetries:   5,
						InitialDelay: 2 * time.Second,
						MaxDelay:     10 * time.Second,
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewOpenAIClient(tt.config)
			if got.config.APIKey != tt.want.config.APIKey {
				t.Errorf("APIKey = %v, want %v", got.config.APIKey, tt.want.config.APIKey)
			}
			if got.config.BaseURL != tt.want.config.BaseURL {
				t.Errorf("BaseURL = %v, want %v", got.config.BaseURL, tt.want.config.BaseURL)
			}
			if got.config.Timeout != tt.want.config.Timeout {
				t.Errorf("Timeout = %v, want %v", got.config.Timeout, tt.want.config.Timeout)
			}
		})
	}
}

func TestOpenAIClient_Complete(t *testing.T) {
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
				"id": "test-id",
				"choices": [
					{
						"message": {
							"content": "Test response"
						},
						"finish_reason": "stop"
					}
				],
				"model": "gpt-4"
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantResp: &CompletionResponse{
				Content:      "Test response",
				Model:        "gpt-4",
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
				if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
					t.Errorf("Path = %v, want /chat/completions", r.URL.Path)
				}
				if r.Header.Get("Authorization") != "Bearer test-key" {
					t.Errorf("Authorization header = %v, want Bearer test-key", r.Header.Get("Authorization"))
				}

				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.response))
			}))
			defer server.Close()

			client := NewOpenAIClient(OpenAIConfig{
				APIKey:  "test-key",
				BaseURL: server.URL,
			})

			got, err := client.Complete(context.Background(), &CompletionRequest{
				Model:  "gpt-4",
				Prompt: "Test prompt",
			})

			if (err != nil) != tt.wantErr {
				t.Errorf("Complete() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got.Content != tt.wantResp.Content {
				t.Errorf("Content = %v, want %v", got.Content, tt.wantResp.Content)
			}
		})
	}
}

func TestOpenAIClient_CompleteStream(t *testing.T) {
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
				"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
				"data: {\"choices\":[{\"delta\":{\"content\":\" World\"}}]}\n\n",
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

			client := NewOpenAIClient(OpenAIConfig{
				APIKey:  "test-key",
				BaseURL: server.URL,
			})

			stream, err := client.CompleteStream(context.Background(), &CompletionRequest{
				Model:  "gpt-4",
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

func TestOpenAIClient_RetryBehavior(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		json.NewEncoder(w).Encode(openaiResponse{
			Choices: []choice{
				{
					Message: openaiMessage{
						Content: "Success after retry",
					},
				},
			},
		})
	}))
	defer server.Close()

	client := NewOpenAIClient(OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
		RetryConfig: &RetryConfig{
			MaxRetries:   3,
			InitialDelay: 10 * time.Millisecond,
			MaxDelay:     50 * time.Millisecond,
			RetryableStatusCodes: []int{
				http.StatusTooManyRequests,
			},
		},
	})

	resp, err := client.Complete(context.Background(), &CompletionRequest{
		Model:  "gpt-4",
		Prompt: "Test prompt",
	})

	if err != nil {
		t.Errorf("Complete() error = %v", err)
		return
	}

	if attempts != 3 {
		t.Errorf("Got %d attempts, want 3", attempts)
	}

	if resp.Content != "Success after retry" {
		t.Errorf("Content = %v, want 'Success after retry'", resp.Content)
	}
}
