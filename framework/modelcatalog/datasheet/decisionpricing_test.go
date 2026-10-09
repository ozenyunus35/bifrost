package datasheet

import (
	"encoding/json"
	"testing"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Rates used across the decision pricing tests. The chat and decision rates
// differ on purpose so a test fails if the wrong set is applied.
const (
	lunaChatInput       = 2e-6
	lunaChatOutput      = 8e-6
	lunaChatAbove272k   = 4e-6
	lunaDecisionInput   = 1e-7
	lunaDecisionOutput  = 3e-7
	decisionTestEpsilon = 1e-12
)

// lunaChatRow builds the gpt-6-luna chat row: chat rates plus a 272k tier, so a
// tier leaking into decision pricing is caught. withDecisionRates adds the
// decision rates the datasheet carries on the same row.
func lunaChatRow(withDecisionRates bool) configstoreTables.TableModelPricing {
	row := configstoreTables.TableModelPricing{
		Model: "gpt-6-luna", Provider: "openai", Mode: "chat",
		InputCostPerToken:                bifrost.Ptr(lunaChatInput),
		OutputCostPerToken:               bifrost.Ptr(lunaChatOutput),
		InputCostPerTokenAbove272kTokens: bifrost.Ptr(lunaChatAbove272k),
	}
	if withDecisionRates {
		row.InputCostPerTokenDecisions = bifrost.Ptr(lunaDecisionInput)
		row.OutputCostPerTokenDecisions = bifrost.Ptr(lunaDecisionOutput)
	}
	return row
}

// lunaStore builds a store holding the given gpt-6-luna chat row.
func lunaStore(row configstoreTables.TableModelPricing) *Store {
	return testStoreWithPricing(map[string]configstoreTables.TableModelPricing{
		makeKey(row.Model, row.Provider, row.Mode): row,
	})
}

// lunaDecisionCost prices a decision on openai/gpt-6-luna with the given usage.
func lunaDecisionCost(s *Store, prompt, completion int) float64 {
	return s.CalculateCost(makeDecisionResponse(schemas.OpenAI, "gpt-6-luna", &schemas.BifrostLLMUsage{
		PromptTokens: prompt, CompletionTokens: completion, TotalTokens: prompt + completion,
	}), nil)
}

// TestDecisionPricing_NativeUsesDecisionRates pins that a chat row carrying
// decision rates prices a decision from them, not from its chat rates.
func TestDecisionPricing_NativeUsesDecisionRates(t *testing.T) {
	s := lunaStore(lunaChatRow(true))

	assert.InDelta(t, 1000*lunaDecisionInput+50*lunaDecisionOutput, lunaDecisionCost(s, 1000, 50), decisionTestEpsilon)
}

// TestDecisionPricing_WithoutDecisionRatesUsesChatRates pins the rule for a
// model with no decision rates, whose decision is a chat call by emulation.
func TestDecisionPricing_WithoutDecisionRatesUsesChatRates(t *testing.T) {
	s := lunaStore(lunaChatRow(false))

	assert.InDelta(t, 1000*lunaChatInput+50*lunaChatOutput, lunaDecisionCost(s, 1000, 50), decisionTestEpsilon)
}

// TestDecisionPricing_DecisionRatesAreFlat pins that a chat tier cannot override
// the flat decision rate on a prompt past the tier's threshold.
func TestDecisionPricing_DecisionRatesAreFlat(t *testing.T) {
	s := lunaStore(lunaChatRow(true))

	assert.InDelta(t, 300000*lunaDecisionInput, lunaDecisionCost(s, 300000, 0), decisionTestEpsilon)
}

// TestDecisionPricing_PartialDecisionRates pins that either decision rate being
// present selects the decision rates, with a missing side billed as zero.
func TestDecisionPricing_PartialDecisionRates(t *testing.T) {
	inputOnly := lunaChatRow(false)
	inputOnly.InputCostPerTokenDecisions = bifrost.Ptr(lunaDecisionInput)
	outputOnly := lunaChatRow(false)
	outputOnly.OutputCostPerTokenDecisions = bifrost.Ptr(lunaDecisionOutput)

	assert.InDelta(t, 1000*lunaDecisionInput, lunaDecisionCost(lunaStore(inputOnly), 1000, 50), decisionTestEpsilon)
	assert.InDelta(t, 50*lunaDecisionOutput, lunaDecisionCost(lunaStore(outputOnly), 1000, 50), decisionTestEpsilon)
}

// TestDecisionPricing_ChatRequestIgnoresDecisionRates pins that decision rates
// never reach a chat request on the same row.
func TestDecisionPricing_ChatRequestIgnoresDecisionRates(t *testing.T) {
	s := lunaStore(lunaChatRow(true))
	resp := makeChatResponse(schemas.OpenAI, "gpt-6-luna", &schemas.BifrostLLMUsage{
		PromptTokens: 1000, CompletionTokens: 50, TotalTokens: 1050,
	})

	assert.InDelta(t, 1000*lunaChatInput+50*lunaChatOutput, s.CalculateCost(resp, nil), decisionTestEpsilon)
}

// setLunaOverride installs a provider-scoped override on openai/gpt-6-luna for
// the given request types, patching the input rate to inputRate.
func setLunaOverride(t *testing.T, s *Store, inputRate string, requestTypes ...schemas.RequestType) {
	t.Helper()
	providerID := "openai"
	require.NoError(t, s.SetOverrides([]configstoreTables.TablePricingOverride{{
		ID:               "luna-override",
		ScopeKind:        string(ScopeKindProvider),
		ProviderID:       &providerID,
		MatchType:        string(MatchTypeExact),
		Pattern:          "gpt-6-luna",
		RequestTypes:     requestTypes,
		PricingPatchJSON: `{"input_cost_per_token":` + inputRate + `}`,
	}}))
}

// TestDecisionPricing_DecisionsOverrideAppliesOverEitherRate pins that an
// override scoped to decisions wins over decision rates and over chat rates.
func TestDecisionPricing_DecisionsOverrideAppliesOverEitherRate(t *testing.T) {
	for name, withDecisionRates := range map[string]bool{"decision rates": true, "chat rates": false} {
		t.Run(name, func(t *testing.T) {
			s := lunaStore(lunaChatRow(withDecisionRates))
			setLunaOverride(t, s, "0.000005", schemas.DecisionRequest)

			outputRate := lunaChatOutput
			if withDecisionRates {
				outputRate = lunaDecisionOutput
			}
			assert.InDelta(t, 1000*0.000005+50*outputRate, lunaDecisionCost(s, 1000, 50), decisionTestEpsilon)
		})
	}
}

// TestDecisionPricing_ChatOverrideDoesNotApply pins that an override scoped to
// chat leaves decision pricing alone.
func TestDecisionPricing_ChatOverrideDoesNotApply(t *testing.T) {
	s := lunaStore(lunaChatRow(true))
	setLunaOverride(t, s, "0.000005", schemas.ChatCompletionRequest)

	assert.InDelta(t, 1000*lunaDecisionInput+50*lunaDecisionOutput, lunaDecisionCost(s, 1000, 50), decisionTestEpsilon)
}

// TestDecisionPricing_DecisionsModeRowWins pins that a decisions-mode row, as
// Typesafe's Jev models have, is priced from itself and never falls to a chat
// row of the same model.
func TestDecisionPricing_DecisionsModeRowWins(t *testing.T) {
	s := testStoreWithPricing(map[string]configstoreTables.TableModelPricing{
		makeKey("jev-1.13.0", "typesafe", "decisions"): {
			Model: "jev-1.13.0", Provider: "typesafe", Mode: "decisions",
			InputCostPerToken: bifrost.Ptr(0.000000042), OutputCostPerToken: bifrost.Ptr(0.0),
		},
		makeKey("jev-1.13.0", "typesafe", "chat"): {
			Model: "jev-1.13.0", Provider: "typesafe", Mode: "chat",
			InputCostPerToken: bifrost.Ptr(1.0), OutputCostPerToken: bifrost.Ptr(1.0),
		},
	})
	resp := makeDecisionResponse(schemas.Typesafe, "jev-1.13.0", &schemas.BifrostLLMUsage{PromptTokens: 50000, TotalTokens: 50000})

	assert.InDelta(t, 50000*0.000000042, s.CalculateCost(resp, nil), decisionTestEpsilon)
}

// TestDecisionPricing_NegativeRateIsUnpriced pins that a variable-price row
// (rates of -1) falling back from a decision is left unpriced, not billed as a
// negative cost.
func TestDecisionPricing_NegativeRateIsUnpriced(t *testing.T) {
	s := testStoreWithPricing(map[string]configstoreTables.TableModelPricing{
		makeKey("typesafe/jev-router", "openrouter", "chat"): {
			Model: "typesafe/jev-router", Provider: "openrouter", Mode: "chat",
			InputCostPerToken: bifrost.Ptr(-1.0), OutputCostPerToken: bifrost.Ptr(-1.0),
		},
	})
	resp := makeDecisionResponse(schemas.OpenRouter, "typesafe/jev-router", &schemas.BifrostLLMUsage{PromptTokens: 1000, CompletionTokens: 10, TotalTokens: 1010})

	assert.Equal(t, 0.0, s.CalculateCost(resp, nil))
}

// TestDecisionPricing_ResponsesOnlyRowStaysUnpriced pins the known gap: a model
// with only a responses row is not priced for decisions.
func TestDecisionPricing_ResponsesOnlyRowStaysUnpriced(t *testing.T) {
	row := lunaChatRow(true)
	row.Mode = "responses"
	s := lunaStore(row)

	assert.Equal(t, 0.0, lunaDecisionCost(s, 1000, 50))
}

// TestDecisionPricing_ProviderFallbacksReachChatRow pins that the Bedrock vendor
// prefix and Vertex prefix-strip lookups retry a decision in chat mode.
func TestDecisionPricing_ProviderFallbacksReachChatRow(t *testing.T) {
	s := testStoreWithPricing(map[string]configstoreTables.TableModelPricing{
		makeKey("anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock", "chat"): {
			Model: "anthropic.claude-3-5-sonnet-20241022-v2:0", Provider: "bedrock", Mode: "chat",
			InputCostPerToken: bifrost.Ptr(3e-6), OutputCostPerToken: bifrost.Ptr(15e-6),
		},
		makeKey("claude-3-5-sonnet", "vertex", "chat"): {
			Model: "claude-3-5-sonnet", Provider: "vertex", Mode: "chat",
			InputCostPerToken: bifrost.Ptr(4e-6), OutputCostPerToken: bifrost.Ptr(20e-6),
		},
	})

	bedrock := s.resolvePricing(schemas.RoutingInfo{Provider: "bedrock", Model: "claude-3-5-sonnet-20241022-v2:0"}, schemas.DecisionRequest, LookupScopes{})
	require.NotNil(t, bedrock)
	assert.Equal(t, 3e-6, *bedrock.InputCostPerToken)

	vertex := s.resolvePricing(schemas.RoutingInfo{Provider: "vertex", Model: "anthropic/claude-3-5-sonnet"}, schemas.DecisionRequest, LookupScopes{})
	require.NotNil(t, vertex)
	assert.Equal(t, 4e-6, *vertex.InputCostPerToken)
}

// TestDecisionRates_LeavesNonDecisionRowsAlone pins that decisionRates returns
// the row unchanged for any other request type and for a row without decision
// rates, and never mutates the row it is given.
func TestDecisionRates_LeavesNonDecisionRowsAlone(t *testing.T) {
	withRates := lunaChatRow(true)

	chat := decisionRates(schemas.ChatCompletionRequest, withRates)
	assert.Equal(t, withRates, chat)

	withoutRates := lunaChatRow(false)
	assert.Equal(t, withoutRates, decisionRates(schemas.DecisionRequest, withoutRates))

	_ = decisionRates(schemas.DecisionRequest, withRates)
	assert.Equal(t, lunaChatInput, *withRates.InputCostPerToken)
	assert.Equal(t, lunaChatOutput, *withRates.OutputCostPerToken)
}

// TestDecisionRates_KeepsRowLevelModifiers pins that the decision row keeps the
// per-request fee and off-peak settings the cost computation reads.
func TestDecisionRates_KeepsRowLevelModifiers(t *testing.T) {
	row := lunaChatRow(true)
	row.CostPerRequest = bifrost.Ptr(0.01)
	row.OffPeakCostMultiplier = bifrost.Ptr(0.5)

	got := decisionRates(schemas.DecisionRequest, row)

	assert.Equal(t, "decisions", got.Mode)
	assert.Equal(t, row.CostPerRequest, got.CostPerRequest)
	assert.Equal(t, row.OffPeakCostMultiplier, got.OffPeakCostMultiplier)
	assert.Nil(t, got.InputCostPerTokenAbove272kTokens)
}

// TestComputeDecisionCost_NegativeRateIsUnpriced pins the guard directly.
func TestComputeDecisionCost_NegativeRateIsUnpriced(t *testing.T) {
	usage := &schemas.BifrostLLMUsage{PromptTokens: 1000, CompletionTokens: 10, TotalTokens: 1010}

	negativeInput := configstoreTables.TableModelPricing{InputCostPerToken: bifrost.Ptr(-1.0), OutputCostPerToken: bifrost.Ptr(0.0)}
	negativeOutput := configstoreTables.TableModelPricing{InputCostPerToken: bifrost.Ptr(1e-6), OutputCostPerToken: bifrost.Ptr(-1.0)}

	assert.Nil(t, computeDecisionCost(&negativeInput, usage, serviceTier{}))
	assert.Nil(t, computeDecisionCost(&negativeOutput, usage, serviceTier{}))
}

// TestDecisionRates_SurviveEntryAndTableConversion pins that the decision rates
// parse from the datasheet JSON and survive both conversions.
func TestDecisionRates_SurviveEntryAndTableConversion(t *testing.T) {
	var entry Entry
	require.NoError(t, json.Unmarshal([]byte(`{
		"provider": "openai", "mode": "chat", "base_model": "gpt-6-luna",
		"input_cost_per_token": 2e-6, "output_cost_per_token": 8e-6,
		"input_cost_per_token_decisions": 1e-7, "output_cost_per_token_decisions": 3e-7
	}`), &entry))
	require.NotNil(t, entry.InputCostPerTokenDecisions)
	require.NotNil(t, entry.OutputCostPerTokenDecisions)

	row := convertEntryToTablePricing("gpt-6-luna", entry)
	assert.Equal(t, 1e-7, *row.InputCostPerTokenDecisions)
	assert.Equal(t, 3e-7, *row.OutputCostPerTokenDecisions)

	back := convertTablePricingToEntry(&row)
	assert.Equal(t, 1e-7, *back.InputCostPerTokenDecisions)
	assert.Equal(t, 3e-7, *back.OutputCostPerTokenDecisions)
}
