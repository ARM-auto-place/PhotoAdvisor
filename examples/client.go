// Package photoadvisor demonstrates the integration contract. Copy into your Go backend.
package photoadvisor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Photo struct {
	PhotoID string `json:"photo_id"`
	ObjectKey string `json:"object_key"`
}

type RPCError struct {
	Code int `json:"code"`
	Message string `json:"message"`
}

type Coverage struct {
	Present []string `json:"present"`
	Missing []string `json:"missing"`
	Uncertain []string `json:"uncertain"`
}

type Result struct {
	ListingID string `json:"listing_id"`
	Status string `json:"status"`
	ProcessingStatus string `json:"processing_status"`
	Required Coverage `json:"required"`
	Recommended Coverage `json:"recommended"`
	Categories []json.RawMessage `json:"categories"`
	PhotoResults []json.RawMessage `json:"photo_results"`
	Suggestions []json.RawMessage `json:"suggestions"`
	Notices []string `json:"notices"`
	Model json.RawMessage `json:"model"`
}

func Evaluate(ctx context.Context, endpoint, token, requestID, listingID string, photos []Photo) (*Result, error) {
	if photos == nil { photos = []Photo{} }
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": requestID, "method": "photos.evaluate",
		"params": map[string]any{"listing_id": listingID, "photos": photos, "locale": "ru"},
	})
	if err != nil { return nil, err }
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil { return nil, err }
	req.Header.Set("Authorization", "Bearer " + token)
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil { return nil, err }
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK { return nil, fmt.Errorf("photo advisor HTTP %d", resp.StatusCode) }
	data, err := io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024+1))
	if err != nil { return nil, err }
	if len(data) > 2*1024*1024 { return nil, fmt.Errorf("photo advisor response too large") }
	var envelope struct {
		JSONRPC string `json:"jsonrpc"`
		ID string `json:"id"`
		Result *Result `json:"result"`
		Error *RPCError `json:"error"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil { return nil, err }
	if envelope.JSONRPC != "2.0" || envelope.ID != requestID { return nil, fmt.Errorf("invalid RPC envelope") }
	if envelope.Error != nil { return nil, fmt.Errorf("photo advisor RPC %d: %s", envelope.Error.Code, envelope.Error.Message) }
	if envelope.Result == nil { return nil, fmt.Errorf("missing RPC result") }
	return envelope.Result, nil
}
