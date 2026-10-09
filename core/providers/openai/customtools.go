package openai

import (
	"slices"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Custom (freeform) tools shipped with GPT-5, and older models reject them with
// "Invalid value: 'custom'" on tools. For those models every custom tool is sent
// as a function tool taking one string parameter, and the model's function call
// is turned back into a custom_tool_call on the way out (Responses API only; the
// Chat schema has no custom tool call shape to restore into).

// customToolInputParam is the single string parameter of a rewritten custom tool.
const customToolInputParam = "input"

// customToolsUnsupported reports whether the target model rejects custom tools.
// Datasheet only: a model is converted only when its row sets
// supports_custom_tools to false, and anything without the flag passes custom
// tools through. Resolved on the base provider so a custom provider built on
// OpenAI gets the same answer as OpenAI itself.
func customToolsUnsupported(ctx *schemas.BifrostContext, provider schemas.ModelProvider, model string) bool {
	capModel := schemas.ResolveCanonicalModel(ctx, model)
	caps := schemas.ResolveModelCaps(schemas.ResolveBaseProvider(ctx, provider), capModel)
	return !caps.SupportsCustomTools(true)
}

// customToolParameters builds the function schema standing in for a custom tool.
// The grammar, when present, goes into the parameter description so the model
// still sees the constraint it would otherwise enforce.
func customToolParameters(syntax, definition string) *schemas.ToolFunctionParameters {
	description := "Raw input for the tool, passed as a plain string."
	if definition != "" {
		if syntax == "" {
			syntax = "grammar"
		}
		description += " It must match this " + syntax + " definition:\n" + definition
	}
	return &schemas.ToolFunctionParameters{
		Type: "object",
		Properties: schemas.NewOrderedMapFromPairs(
			schemas.KV(customToolInputParam, schemas.NewOrderedMapFromPairs(
				schemas.KV("type", "string"),
				schemas.KV("description", description),
			)),
		),
		Required:             []string{customToolInputParam},
		AdditionalProperties: &schemas.AdditionalPropertiesStruct{AdditionalPropertiesBool: schemas.Ptr(false)},
	}
}

// wrapCustomToolInput encodes a custom tool input as function call arguments.
func wrapCustomToolInput(input string) string {
	encoded, err := sonic.MarshalString(input)
	if err != nil {
		return `{"` + customToolInputParam + `":""}`
	}
	return `{"` + customToolInputParam + `":` + encoded + `}`
}

// unwrapCustomToolInput extracts the custom tool input from function call
// arguments. Arguments that are not the expected object are returned as-is so the
// model's output is never lost.
func unwrapCustomToolInput(arguments string) string {
	field := providerUtils.GetJSONField([]byte(arguments), customToolInputParam)
	if !field.Exists() {
		return arguments
	}
	return field.String()
}

// responsesCustomToolsAsFunctions rewrites custom tools as function tools.
// Returns the input slice untouched when there is nothing to rewrite.
func responsesCustomToolsAsFunctions(tools []schemas.ResponsesTool) []schemas.ResponsesTool {
	if !containsResponsesCustomTool(tools) {
		return tools
	}
	rewritten := make([]schemas.ResponsesTool, len(tools))
	for i, tool := range tools {
		if tool.Type == schemas.ResponsesToolTypeCustom {
			var syntax, definition string
			if tool.ResponsesToolCustom != nil && tool.ResponsesToolCustom.Format != nil {
				if f := tool.ResponsesToolCustom.Format; f.Definition != nil {
					definition = *f.Definition
					if f.Syntax != nil {
						syntax = *f.Syntax
					}
				}
			}
			tool.Type = schemas.ResponsesToolTypeFunction
			tool.ResponsesToolCustom = nil
			tool.ResponsesToolFunction = &schemas.ResponsesToolFunction{
				Parameters: customToolParameters(syntax, definition),
				Strict:     schemas.Ptr(false),
			}
		}
		rewritten[i] = tool
	}
	return rewritten
}

func containsResponsesCustomTool(tools []schemas.ResponsesTool) bool {
	for _, tool := range tools {
		if tool.Type == schemas.ResponsesToolTypeCustom {
			return true
		}
	}
	return false
}

// responsesCustomToolChoiceAsFunction points a custom tool choice, or the custom
// entries of an allowed_tools choice, at the rewritten function tool.
func responsesCustomToolChoiceAsFunction(choice *schemas.ResponsesToolChoice) *schemas.ResponsesToolChoice {
	if choice == nil || choice.ResponsesToolChoiceStruct == nil {
		return choice
	}
	s := choice.ResponsesToolChoiceStruct
	if s.Type == schemas.ResponsesToolChoiceTypeCustom {
		structCopy := *s
		structCopy.Type = schemas.ResponsesToolChoiceTypeFunction
		return &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &structCopy}
	}
	if s.Type == schemas.ResponsesToolChoiceTypeAllowedTools {
		var tools []schemas.ResponsesToolChoiceAllowedToolDef
		for i, tool := range s.Tools {
			if tool.Type != string(schemas.ResponsesToolTypeCustom) {
				continue
			}
			if tools == nil {
				tools = slices.Clone(s.Tools)
			}
			tools[i].Type = string(schemas.ResponsesToolTypeFunction)
		}
		if tools != nil {
			structCopy := *s
			structCopy.Tools = tools
			return &schemas.ResponsesToolChoice{ResponsesToolChoiceStruct: &structCopy}
		}
	}
	return choice
}

// responsesCustomCallItemsAsFunctions rewrites replayed custom_tool_call and
// custom_tool_call_output items as their function equivalents. Rewritten items
// get a fresh ResponsesToolMessage so the caller's input is never mutated.
func responsesCustomCallItemsAsFunctions(messages []schemas.ResponsesMessage) []schemas.ResponsesMessage {
	for i := range messages {
		message := &messages[i]
		if message.Type == nil || message.ResponsesToolMessage == nil {
			continue
		}
		switch *message.Type {
		case schemas.ResponsesMessageTypeCustomToolCall:
			toolMessage := *message.ResponsesToolMessage
			input := ""
			if toolMessage.ResponsesCustomToolCall != nil {
				input = toolMessage.ResponsesCustomToolCall.Input
			}
			toolMessage.ResponsesCustomToolCall = nil
			toolMessage.Arguments = schemas.Ptr(wrapCustomToolInput(input))
			message.ResponsesToolMessage = &toolMessage
			message.Type = schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall)
		case schemas.ResponsesMessageTypeCustomToolCallOutput:
			message.Type = schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput)
		}
	}
	return messages
}

// customToolNamesToRestore returns the names of the custom tools this request
// sent as function tools, or nil when none were rewritten. A raw request body
// is forwarded unconverted, so nothing is restored for it either.
func customToolNamesToRestore(ctx *schemas.BifrostContext, request *schemas.BifrostResponsesRequest) map[string]struct{} {
	if request == nil || request.Params == nil || !containsResponsesCustomTool(request.Params.Tools) {
		return nil
	}
	if ctx != nil {
		if _, raw := providerUtils.CheckAndGetRawRequestBody(ctx, request); raw {
			return nil
		}
	}
	if !customToolsUnsupported(ctx, request.Provider, request.Model) {
		return nil
	}
	names := make(map[string]struct{})
	for _, tool := range request.Params.Tools {
		if tool.Type == schemas.ResponsesToolTypeCustom && tool.Name != nil {
			names[*tool.Name] = struct{}{}
		}
	}
	return names
}

// restoreCustomToolCall turns a function_call item for a rewritten custom tool
// back into a custom_tool_call. Reports whether the item was rewritten.
func restoreCustomToolCall(item *schemas.ResponsesMessage, names map[string]struct{}) bool {
	if item == nil || item.Type == nil || *item.Type != schemas.ResponsesMessageTypeFunctionCall ||
		item.ResponsesToolMessage == nil || item.ResponsesToolMessage.Name == nil {
		return false
	}
	if _, ok := names[*item.ResponsesToolMessage.Name]; !ok {
		return false
	}
	toolMessage := *item.ResponsesToolMessage
	input := ""
	if toolMessage.Arguments != nil {
		input = unwrapCustomToolInput(*toolMessage.Arguments)
	}
	toolMessage.Arguments = nil
	toolMessage.ResponsesCustomToolCall = &schemas.ResponsesCustomToolCall{Input: input}
	item.ResponsesToolMessage = &toolMessage
	item.Type = schemas.Ptr(schemas.ResponsesMessageTypeCustomToolCall)
	return true
}

// restoreCustomToolCalls applies restoreCustomToolCall to every output item.
func restoreCustomToolCalls(output []schemas.ResponsesMessage, names map[string]struct{}) {
	for i := range output {
		restoreCustomToolCall(&output[i], names)
	}
}

// customToolStreamRestorer restores custom tool calls across one Responses stream.
// Argument deltas of a rewritten call are held back because partial JSON cannot
// be unwrapped; the full input is emitted once the arguments are done.
type customToolStreamRestorer struct {
	names   map[string]struct{}
	itemIDs map[string]struct{}
	lastSeq map[string]int // sequence number of the last held-back delta, per item
}

func newCustomToolStreamRestorer(names map[string]struct{}) *customToolStreamRestorer {
	if len(names) == 0 {
		return nil
	}
	return &customToolStreamRestorer{names: names, itemIDs: map[string]struct{}{}, lastSeq: map[string]int{}}
}

// restore rewrites one stream event in place. drop reports that the event must
// not be sent; prefix, when non-nil, must be sent just before it.
func (r *customToolStreamRestorer) restore(event *schemas.BifrostResponsesStreamResponse) (prefix *schemas.BifrostResponsesStreamResponse, drop bool) {
	if r == nil || event == nil {
		return nil, false
	}
	switch event.Type {
	case schemas.ResponsesStreamResponseTypeOutputItemAdded, schemas.ResponsesStreamResponseTypeOutputItemDone:
		if restoreCustomToolCall(event.Item, r.names) && event.Item.ID != nil {
			r.itemIDs[*event.Item.ID] = struct{}{}
		}
	case schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta:
		if r.tracked(event.ItemID) {
			r.lastSeq[*event.ItemID] = event.SequenceNumber
			return nil, true
		}
	case schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDone:
		if !r.tracked(event.ItemID) {
			return nil, false
		}
		input := ""
		if event.Arguments != nil {
			input = unwrapCustomToolInput(*event.Arguments)
		}
		seq, ok := r.lastSeq[*event.ItemID]
		if !ok {
			seq = event.SequenceNumber
		}
		prefix = &schemas.BifrostResponsesStreamResponse{
			Type:           schemas.ResponsesStreamResponseTypeCustomToolCallInputDelta,
			SequenceNumber: seq,
			OutputIndex:    event.OutputIndex,
			ItemID:         event.ItemID,
			Delta:          schemas.Ptr(input),
		}
		event.Type = schemas.ResponsesStreamResponseTypeCustomToolCallInputDone
		event.Arguments = nil
		event.Input = schemas.Ptr(input)
		return prefix, false
	case schemas.ResponsesStreamResponseTypeCompleted, schemas.ResponsesStreamResponseTypeIncomplete:
		if event.Response != nil {
			restoreCustomToolCalls(event.Response.Output, r.names)
		}
	}
	return nil, false
}

func (r *customToolStreamRestorer) tracked(itemID *string) bool {
	if itemID == nil {
		return false
	}
	_, ok := r.itemIDs[*itemID]
	return ok
}

// chatCustomToolAsFunction rewrites a Chat custom tool as a function tool. Chat
// custom tools carry no description, so the function has none either.
func chatCustomToolAsFunction(tool schemas.ChatTool) schemas.ChatTool {
	var syntax, definition string
	if tool.Custom != nil && tool.Custom.Format != nil && tool.Custom.Format.Grammar != nil {
		syntax = tool.Custom.Format.Grammar.Syntax
		definition = tool.Custom.Format.Grammar.Definition
	}
	tool.Type = schemas.ChatToolTypeFunction
	tool.Function = &schemas.ChatToolFunction{
		Name:       tool.Name,
		Parameters: customToolParameters(syntax, definition),
	}
	tool.Custom = nil
	tool.Name = ""
	// The by-value copy may carry a precomputed serialized cache of the custom shape.
	tool.InvalidateSerialized()
	return tool
}

// chatCustomToolChoiceAsFunction points a custom tool choice at the rewritten
// function tool.
func chatCustomToolChoiceAsFunction(choice *schemas.ChatToolChoice) *schemas.ChatToolChoice {
	if choice == nil || choice.ChatToolChoiceStruct == nil ||
		choice.ChatToolChoiceStruct.Type != schemas.ChatToolChoiceTypeCustom {
		return choice
	}
	name := ""
	if choice.ChatToolChoiceStruct.Custom != nil {
		name = choice.ChatToolChoiceStruct.Custom.Name
	}
	return &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
		Type:     schemas.ChatToolChoiceTypeFunction,
		Function: &schemas.ChatToolChoiceFunction{Name: name},
	}}
}
