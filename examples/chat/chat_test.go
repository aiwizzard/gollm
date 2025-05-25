package chat

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/aiwizzard/gollm/llm"
)

func TestRunBasicCompletion(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != http.MethodPost {
			t.Errorf("Method = %v, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("Path = %v, want /chat/completions", r.URL.Path)
		}

		// Return a mock response
		w.Write([]byte(`{
			"choices": [
				{
					"message": {
						"content": "Quantum computing uses quantum mechanics to perform complex calculations exponentially faster than classical computers."
					},
					"finish_reason": "stop"
				}
			],
			"model": "gpt-4"
		}`))
	}))
	defer server.Close()

	// Create a client with the mock server
	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	// Run the example
	err := runBasicCompletion(client)
	if err != nil {
		t.Errorf("runBasicCompletion() error = %v", err)
	}
}

func TestRunStreamingCompletion(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != http.MethodPost {
			t.Errorf("Method = %v, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("Path = %v, want /chat/completions", r.URL.Path)
		}

		// Return a mock streaming response
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("Streaming not supported")
		}

		responses := []string{
			"data: {\"choices\":[{\"delta\":{\"content\":\"Code flows like water\"}}]}\n\n",
			"data: {\"choices\":[{\"delta\":{\"content\":\"\\nBugs hide in shadows\"}}]}\n\n",
			"data: {\"choices\":[{\"delta\":{\"content\":\"\\nDebugger brings light\"}}]}\n\n",
			"data: [DONE]\n\n",
		}

		for _, resp := range responses {
			_, err := w.Write([]byte(resp))
			if err != nil {
				t.Fatal(err)
			}
			flusher.Flush()
		}
	}))
	defer server.Close()

	// Create a client with the mock server
	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	// Run the example
	err := runStreamingCompletion(client)
	if err != nil {
		t.Errorf("runStreamingCompletion() error = %v", err)
	}
}

func TestRunCustomizedCompletion(t *testing.T) {
	// Create a mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != http.MethodPost {
			t.Errorf("Method = %v, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("Path = %v, want /chat/completions", r.URL.Path)
		}

		// Return a mock response
		w.Write([]byte(`{
			"choices": [
				{
					"message": {
						"content": "QuantumLeap"
					},
					"finish_reason": "stop"
				}
			],
			"model": "gpt-4"
		}`))
	}))
	defer server.Close()

	// Create a client with the mock server
	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	// Run the example
	err := runCustomizedCompletion(client)
	if err != nil {
		t.Errorf("runCustomizedCompletion() error = %v", err)
	}
}

func TestRunExample(t *testing.T) {
	// Save original env and restore after test
	originalAPIKey := os.Getenv("OPENAI_API_KEY")
	defer os.Setenv("OPENAI_API_KEY", originalAPIKey)

	tests := []struct {
		name    string
		apiKey  string
		wantErr bool
	}{
		{
			name:    "missing API key",
			apiKey:  "",
			wantErr: true,
		},
		{
			name:    "with API key",
			apiKey:  "test-key",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set test API key
			os.Setenv("OPENAI_API_KEY", tt.apiKey)

			// Create a mock server if API key is provided
			var server *httptest.Server
			if tt.apiKey != "" {
				server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Write([]byte(`{
						"choices": [
							{
								"message": {
									"content": "Test response"
								},
								"finish_reason": "stop"
							}
						],
						"model": "gpt-4"
					}`))
				}))
				defer server.Close()

				// Override the default base URL
				os.Setenv("OPENAI_API_BASE_URL", server.URL)
			}

			err := RunExample()
			if (err != nil) != tt.wantErr {
				t.Errorf("RunExample() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
