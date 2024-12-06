using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Services;
using System.Text;

namespace Microsoft.SemanticKernel;

/// <summary>
/// Represents a chat completion service using OnnxRuntimeGenAI.
/// </summary>
public sealed class OnnxRuntimeGenAIChatCompletionService : IChatCompletionService
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly TokenizerStream _tokenizerStream;

    private Dictionary<string, object?> AttributesInternal { get; } = new();

    /// <summary>
    /// Initializes a new instance of the OnnxRuntimeGenAIChatCompletionService class.
    /// </summary>
    /// <param name="modelPath">The generative AI ONNX model path for the chat completion service.</param>
    /// <param name="loggerFactory">Optional logger factory to be used for logging.</param>
    public OnnxRuntimeGenAIChatCompletionService(
        string modelPath,
        ILoggerFactory? loggerFactory = null)
    {
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
        _tokenizerStream = _tokenizer.CreateStream();

        this.AttributesInternal.Add(AIServiceExtensions.ModelIdKey, _tokenizer);
    }

    public IReadOnlyDictionary<string, object?> Attributes => this.AttributesInternal;

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var result = new StringBuilder();

        await foreach (var content in RunInferenceAsync(chatHistory, executionSettings, cancellationToken))
        {
            result.Append(content);
        }

        return new List<ChatMessageContent>
        {
            new(
                role: AuthorRole.Assistant,
                content: result.ToString())
        };
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        await foreach (var content in RunInferenceAsync(chatHistory, executionSettings, cancellationToken))
        {
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, content);
        }
    }

    private async IAsyncEnumerable<string> RunInferenceAsync(ChatHistory chatHistory, PromptExecutionSettings? executionSettings, CancellationToken cancellationToken)
    {
        OnnxRuntimeGenAIPromptExecutionSettings settings = OnnxRuntimeGenAIPromptExecutionSettings.FromExecutionSettings(executionSettings);

        var prompt = GetPrompt(chatHistory);
        var tokens = _tokenizer.Encode(prompt);

        var generatorParams = new GeneratorParams(_model);
        ApplyPromptExecutionSettings(generatorParams, settings);
        generatorParams.SetInputSequences(tokens);

        var generator = new Generator(_model, generatorParams);

        while (!generator.IsDone())
        {
            cancellationToken.ThrowIfCancellationRequested();

            yield return await Task.Run(() =>
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();

                var outputTokens = generator.GetSequence(0);

                // Fix garbage ending token
                var newToken = outputTokens[^1];
                var output = _tokenizerStream.Decode(newToken);

                //var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
                //var output = _tokenizer.Decode(newToken);

                return output;
            }, cancellationToken);
        }
    }

    private string GetPrompt(ChatHistory chatHistory)
    {
        var promptBuilder = new StringBuilder();
        foreach (var message in chatHistory)
        {
            promptBuilder.Append($"<|{message.Role}|>\n{message.Content}");
        }
        promptBuilder.Append($"<|end|>\n<|assistant|>");

        return promptBuilder.ToString();
    }

    private void ApplyPromptExecutionSettings(GeneratorParams generatorParams, OnnxRuntimeGenAIPromptExecutionSettings settings)
    {
        generatorParams.SetSearchOption("top_p", settings.TopP);
        generatorParams.SetSearchOption("top_k", settings.TopK);
        generatorParams.SetSearchOption("temperature", settings.Temperature);
        generatorParams.SetSearchOption("repetition_penalty", settings.RepetitionPenalty);
        generatorParams.SetSearchOption("past_present_share_buffer", settings.PastPresentShareBuffer);
        generatorParams.SetSearchOption("num_return_sequences", settings.NumReturnSequences);
        generatorParams.SetSearchOption("no_repeat_ngram_size", settings.NoRepeatNgramSize);
        generatorParams.SetSearchOption("min_length", settings.MinLength);
        generatorParams.SetSearchOption("max_length", settings.MaxLength);
        generatorParams.SetSearchOption("length_penalty", settings.LengthPenalty);
        generatorParams.SetSearchOption("early_stopping", settings.EarlyStopping);
        generatorParams.SetSearchOption("do_sample", settings.DoSample);
        generatorParams.SetSearchOption("diversity_penalty", settings.DiversityPenalty);
    }
}