package tools

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/aiwizzard/gollm/llm"
)

func TestGetWeather(t *testing.T) {
	tests := []struct {
		name     string
		location string
		unit     string
		want     string
	}{
		{
			name:     "celsius",
			location: "London",
			unit:     "C",
			want:     "The weather in London is 22°C",
		},
		{
			name:     "fahrenheit",
			location: "New York",
			unit:     "F",
			want:     "The weather in New York is 22°F",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := GetWeather(tt.location, tt.unit)
			if got != tt.want {
				t.Errorf("GetWeather() = %v, want %v", got, tt.want)
			}
		})
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

			if tt.apiKey != "" {
				// Create a mock server that simulates the tool calling flow
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("Method = %v, want POST", r.Method)
					}

					// First response with tool call
					if !strings.Contains(r.URL.Path, "completions") {
						t.Errorf("Path = %v, want /chat/completions", r.URL.Path)
					}

					// Parse the request to check if it's the first or second call
					var req struct {
						Messages []struct {
							Content string `json:"content"`
						} `json:"messages"`
					}
					json.NewDecoder(r.Body).Decode(&req)

					if strings.Contains(req.Messages[0].Content, "weather function returned") {
						// Second call - return final response
						w.Write([]byte(`{
							"choices": [
								{
									"message": {
										"content": "Based on the weather data, it's 22°C in London today."
									},
									"finish_reason": "stop"
								}
							]
						}`))
					} else {
						// First call - return tool call
						w.Write([]byte(`{
							"choices": [
								{
									"message": {
										"tool_calls": [
											{
												"id": "call_123",
												"type": "function",
												"function": {
													"name": "get_weather",
													"arguments": "{\"location\":\"London\",\"unit\":\"C\"}"
												}
											}
										]
									},
									"finish_reason": "tool_calls"
								}
							]
						}`))
					}
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

func TestRunNonStreamingExample(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// First response with tool call
		w.Write([]byte(`{
			"choices": [
				{
					"message": {
						"tool_calls": [
							{
								"id": "call_123",
								"type": "function",
								"function": {
									"name": "get_weather",
									"arguments": "{\"location\":\"London\",\"unit\":\"C\"}"
								}
							}
						]
					},
					"finish_reason": "tool_calls"
				}
			]
		}`))
	}))
	defer server.Close()

	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

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

	err := runNonStreamingExample(client, weatherTool)
	if err != nil {
		t.Errorf("runNonStreamingExample() error = %v", err)
	}
}

func TestRunStreamingExample(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("Streaming not supported")
		}

		responses := []string{
			`data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"get_weather","arguments":"{\"location\":\"Paris\",\"unit\":\"F\"}"}}]}}]}`,
			"data: [DONE]\n\n",
		}

		for _, resp := range responses {
			w.Write([]byte(resp + "\n\n"))
			flusher.Flush()
		}
	}))
	defer server.Close()

	client := llm.NewOpenAIClient(llm.OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

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

	err := runStreamingExample(client, weatherTool)
	if err != nil {
		t.Errorf("runStreamingExample() error = %v", err)
	}
}
