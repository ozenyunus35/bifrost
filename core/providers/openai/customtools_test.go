package openai

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

func customResponsesTool(name string) schemas.ResponsesTool {
	return schemas.ResponsesTool{
		Type:        schemas.ResponsesToolTypeCustom,
		Name:        schemas.Ptr(name),
		Description: schemas.Ptr("apply a patch"),
		ResponsesToolCustom: &schemas.ResponsesToolCustom{Format: &schemas.ResponsesToolCustomFormat{
			Type:       "grammar",
			Syntax:     schemas.Ptr("lark"),
			Definition: schemas.Ptr("start: /.+/"),
		}},
	}
}

func customToolsResponsesRequest(model string) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: schemas.OpenAI,
		Model:    model,
		Input: []schemas.ResponsesMessage{
			{
				Type: schemas.Ptr(schemas.ResponsesMessageTypeCustomToolCall),
				ResponsesToolMessage: &schemas.ResponsesToolMessage{
					CallID:                  schemas.Ptr("call_1"),
					Name:                    schemas.Ptr("apply_patch"),
					ResponsesCustomToolCall: &schemas.ResponsesCustomToolCall{Input: "*** Begin Patch \"x\""},
				},
			},
			{
				Type: schemas.Ptr(schemas.ResponsesMessageTypeCustomToolCallOutput),
				ResponsesToolMessage: &schemas.ResponsesToolMessage{
					CallID: schemas.Ptr("call_1"),
					Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("done")},
				},
			},
		},
		Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{customResponsesTool("apply_patch")},
			ToolChoice: &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &schemas.ResponsesToolChoiceStruct{
				Type: schemas.ResponsesToolChoiceTypeCustom,
				Name: schemas.Ptr("apply_patch"),
			}},
		},
	}
}

// stubCustomToolsDatasheet marks gpt-4o as rejecting custom tools and gpt-5 as
// accepting them; every other model has no row.
func stubCustomToolsDatasheet(t *testing.T) {
	rows := map[string]bool{"gpt-4o": false, "gpt-5": true}
	schemas.SetCapabilityResolver(func(_ schemas.ModelProvider, model string) *schemas.ModelCapabilities {
		supported, ok := rows[model]
		if !ok {
			return nil
		}
		return &schemas.ModelCapabilities{SupportsCustomTools: schemas.Ptr(supported)}
	})
	t.Cleanup(func() { schemas.SetCapabilityResolver(nil) })
}

// Older models reject custom tools with "Invalid value: 'custom'"; a row marking
// that sends them as function tools, with replayed calls and the tool choice following.
func TestToOpenAIResponsesRequest_CustomToolsAsFunctionsForOldModels(t *testing.T) {
	stubCustomToolsDatasheet(t)
	bifrostReq := customToolsResponsesRequest("gpt-4o")
	result := ToOpenAIResponsesRequest(nil, bifrostReq)
	require.NotNil(t, result)

	require.Len(t, result.Tools, 1)
	tool := result.Tools[0]
	require.Equal(t, schemas.ResponsesToolTypeFunction, tool.Type)
	require.Nil(t, tool.ResponsesToolCustom)
	require.Equal(t, "apply_patch", *tool.Name)
	require.Equal(t, "apply a patch", *tool.Description)
	require.NotNil(t, tool.ResponsesToolFunction)
	require.Equal(t, []string{customToolInputParam}, tool.ResponsesToolFunction.Parameters.Required)
	input, ok := tool.ResponsesToolFunction.Parameters.Properties.Get(customToolInputParam)
	require.True(t, ok)
	inputSchema := input.(*schemas.OrderedMap)
	description, _ := inputSchema.Get("description")
	require.Contains(t, description, "lark")
	require.Contains(t, description, "start: /.+/")

	choice := result.ToolChoice.ResponsesToolChoiceStruct
	require.Equal(t, schemas.ResponsesToolChoiceTypeFunction, choice.Type)
	require.Equal(t, "apply_patch", *choice.Name)

	items := result.Input.OpenAIResponsesRequestInputArray
	require.Len(t, items, 2)
	require.Equal(t, schemas.ResponsesMessageTypeFunctionCall, *items[0].Type)
	require.Nil(t, items[0].ResponsesCustomToolCall)
	require.JSONEq(t, `{"input":"*** Begin Patch \"x\""}`, *items[0].Arguments)
	require.Equal(t, schemas.ResponsesMessageTypeFunctionCallOutput, *items[1].Type)

	// The caller's request is untouched, since fallbacks re-run the conversion.
	require.Equal(t, schemas.ResponsesToolTypeCustom, bifrostReq.Params.Tools[0].Type)
	require.Equal(t, schemas.ResponsesToolChoiceTypeCustom, bifrostReq.Params.ToolChoice.ResponsesToolChoiceStruct.Type)
	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *bifrostReq.Input[0].Type)
	require.NotNil(t, bifrostReq.Input[0].ResponsesCustomToolCall)
	require.Nil(t, bifrostReq.Input[0].Arguments)
	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCallOutput, *bifrostReq.Input[1].Type)
}

// Custom tools pass through when the row says they are supported, and when there
// is no row or no flag at all: there is no name-based fallback.
func TestToOpenAIResponsesRequest_CustomToolsPassThroughWithoutFalseFlag(t *testing.T) {
	stubCustomToolsDatasheet(t)
	for _, model := range []string{"gpt-5", "gpt-4.1-mini"} {
		result := ToOpenAIResponsesRequest(nil, customToolsResponsesRequest(model))
		require.NotNil(t, result, model)
		require.Equal(t, schemas.ResponsesToolTypeCustom, result.Tools[0].Type, model)
		require.NotNil(t, result.Tools[0].ResponsesToolCustom, model)
		require.Equal(t, schemas.ResponsesToolChoiceTypeCustom, result.ToolChoice.ResponsesToolChoiceStruct.Type, model)
		require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *result.Input.OpenAIResponsesRequestInputArray[0].Type, model)
	}
}

func TestResponsesCustomToolChoiceAsFunction_AllowedTools(t *testing.T) {
	original := &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &schemas.ResponsesToolChoiceStruct{
		Type: schemas.ResponsesToolChoiceTypeAllowedTools,
		Mode: schemas.Ptr("required"),
		Tools: []schemas.ResponsesToolChoiceAllowedToolDef{
			{Type: "custom", Name: schemas.Ptr("apply_patch")},
			{Type: "function", Name: schemas.Ptr("search")},
		},
	}}
	got := responsesCustomToolChoiceAsFunction(original)
	require.Equal(t, "function", got.ResponsesToolChoiceStruct.Tools[0].Type)
	require.Equal(t, "function", got.ResponsesToolChoiceStruct.Tools[1].Type)
	require.Equal(t, "custom", original.ResponsesToolChoiceStruct.Tools[0].Type)
}

func TestCustomToolNamesToRestore(t *testing.T) {
	stubCustomToolsDatasheet(t)
	require.Equal(t, map[string]struct{}{"apply_patch": {}}, customToolNamesToRestore(nil, customToolsResponsesRequest("gpt-4o")))
	require.Nil(t, customToolNamesToRestore(nil, customToolsResponsesRequest("gpt-5")))

	ctx := schemas.NewBifrostContext(t.Context(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
	require.Nil(t, customToolNamesToRestore(ctx, customToolsResponsesRequest("gpt-4o")))
}

func TestRestoreCustomToolCalls(t *testing.T) {
	names := map[string]struct{}{"apply_patch": {}}
	output := []schemas.ResponsesMessage{
		{
			Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID:    schemas.Ptr("call_1"),
				Name:      schemas.Ptr("apply_patch"),
				Arguments: schemas.Ptr(`{"input":"line one\nline two"}`),
			},
		},
		{
			Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				Name:      schemas.Ptr("apply_patch"),
				Arguments: schemas.Ptr(`not json`),
			},
		},
		{
			Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				Name:      schemas.Ptr("search"),
				Arguments: schemas.Ptr(`{"q":"x"}`),
			},
		},
	}
	restoreCustomToolCalls(output, names)

	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *output[0].Type)
	require.Nil(t, output[0].Arguments)
	require.Equal(t, "call_1", *output[0].CallID)
	require.Equal(t, "line one\nline two", output[0].ResponsesCustomToolCall.Input)

	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *output[1].Type)
	require.Equal(t, "not json", output[1].ResponsesCustomToolCall.Input)

	require.Equal(t, schemas.ResponsesMessageTypeFunctionCall, *output[2].Type)
	require.Equal(t, `{"q":"x"}`, *output[2].Arguments)
}

func TestCustomToolStreamRestorer(t *testing.T) {
	require.Nil(t, newCustomToolStreamRestorer(nil))

	r := newCustomToolStreamRestorer(map[string]struct{}{"apply_patch": {}})
	itemID := schemas.Ptr("fc_1")
	outputIndex := schemas.Ptr(0)

	added := &schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
		OutputIndex: outputIndex,
		Item: &schemas.ResponsesMessage{
			ID:                   itemID,
			Type:                 schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{Name: schemas.Ptr("apply_patch"), Arguments: schemas.Ptr("")},
		},
	}
	prefix, drop := r.restore(added)
	require.Nil(t, prefix)
	require.False(t, drop)
	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *added.Item.Type)

	for seq, chunk := range []string{`{"inp`, `ut":"abc"}`} {
		prefix, drop = r.restore(&schemas.BifrostResponsesStreamResponse{
			Type:           schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
			SequenceNumber: seq + 2,
			ItemID:         itemID,
			Delta:          schemas.Ptr(chunk),
		})
		require.Nil(t, prefix)
		require.True(t, drop)
	}

	// Deltas of an unrelated function call pass through.
	_, drop = r.restore(&schemas.BifrostResponsesStreamResponse{
		Type:   schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
		ItemID: schemas.Ptr("fc_other"),
		Delta:  schemas.Ptr("{}"),
	})
	require.False(t, drop)

	done := &schemas.BifrostResponsesStreamResponse{
		Type:           schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDone,
		SequenceNumber: 4,
		OutputIndex:    outputIndex,
		ItemID:         itemID,
		Arguments:      schemas.Ptr(`{"input":"abc"}`),
	}
	prefix, drop = r.restore(done)
	require.False(t, drop)
	require.NotNil(t, prefix)
	require.Equal(t, schemas.ResponsesStreamResponseTypeCustomToolCallInputDelta, prefix.Type)
	require.Equal(t, 3, prefix.SequenceNumber)
	require.Equal(t, "abc", *prefix.Delta)
	require.Equal(t, schemas.ResponsesStreamResponseTypeCustomToolCallInputDone, done.Type)
	require.Nil(t, done.Arguments)
	require.Equal(t, "abc", *done.Input)

	completed := &schemas.BifrostResponsesStreamResponse{
		Type: schemas.ResponsesStreamResponseTypeCompleted,
		Response: &schemas.BifrostResponsesResponse{Output: []schemas.ResponsesMessage{{
			ID:                   itemID,
			Type:                 schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{Name: schemas.Ptr("apply_patch"), Arguments: schemas.Ptr(`{"input":"abc"}`)},
		}}},
	}
	r.restore(completed)
	require.Equal(t, schemas.ResponsesMessageTypeCustomToolCall, *completed.Response.Output[0].Type)
	require.Equal(t, "abc", completed.Response.Output[0].ResponsesCustomToolCall.Input)
}

func customChatRequest(model string) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    model,
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ChatParameters{
			Tools: []schemas.ChatTool{{
				Type: schemas.ChatToolTypeCustom,
				Name: "apply_patch",
				Custom: &schemas.ChatToolCustom{Format: &schemas.ChatToolCustomFormat{
					Type:    "grammar",
					Grammar: &schemas.ChatToolCustomGrammarFormat{Syntax: "regex", Definition: "^\\d+$"},
				}},
			}},
			ToolChoice: &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
				Type:   schemas.ChatToolChoiceTypeCustom,
				Custom: &schemas.ChatToolChoiceCustom{Name: "apply_patch"},
			}},
		},
	}
}

func TestToOpenAIChatRequest_CustomToolsAsFunctionsForOldModels(t *testing.T) {
	stubCustomToolsDatasheet(t)
	bifrostReq := customChatRequest("gpt-4o")
	result := ToOpenAIChatRequest(nil, bifrostReq)
	require.NotNil(t, result)

	tool := result.ChatParameters.Tools[0]
	require.Equal(t, schemas.ChatToolTypeFunction, tool.Type)
	require.Nil(t, tool.Custom)
	require.Equal(t, "apply_patch", tool.Function.Name)
	input, ok := tool.Function.Parameters.Properties.Get(customToolInputParam)
	require.True(t, ok)
	description, _ := input.(*schemas.OrderedMap).Get("description")
	require.Contains(t, description, "^\\d+$")

	choice := result.ChatParameters.ToolChoice.ChatToolChoiceStruct
	require.Equal(t, schemas.ChatToolChoiceTypeFunction, choice.Type)
	require.Equal(t, "apply_patch", choice.Function.Name)

	body, err := schemas.MarshalSorted(tool)
	require.NoError(t, err)
	require.NotContains(t, string(body), `"custom"`)

	require.Equal(t, schemas.ChatToolTypeCustom, bifrostReq.Params.Tools[0].Type)
	require.Equal(t, schemas.ChatToolChoiceTypeCustom, bifrostReq.Params.ToolChoice.ChatToolChoiceStruct.Type)
}

func TestToOpenAIChatRequest_CustomToolsPassThroughWithoutFalseFlag(t *testing.T) {
	stubCustomToolsDatasheet(t)
	for _, model := range []string{"gpt-5", "gpt-4.1-mini"} {
		result := ToOpenAIChatRequest(nil, customChatRequest(model))
		require.NotNil(t, result, model)
		require.Equal(t, schemas.ChatToolTypeCustom, result.ChatParameters.Tools[0].Type, model)
		require.Equal(t, schemas.ChatToolChoiceTypeCustom, result.ChatParameters.ToolChoice.ChatToolChoiceStruct.Type, model)
	}
}
