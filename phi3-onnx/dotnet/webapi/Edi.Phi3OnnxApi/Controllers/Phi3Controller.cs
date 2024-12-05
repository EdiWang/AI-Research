using Edi.Phi3OnnxApi.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Edi.Phi3OnnxApi.Controllers;

[ApiController]
[Route("[controller]")]
public class Phi3Controller : Controller
{
    private readonly ILogger<Phi3Controller> _logger;
    private readonly IConfiguration _configuration;

    private static Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly TokenizerStream _tokenizerStream;
    private static string _defaultSystemPrompt;

    public Phi3Controller(IConfiguration configuration, ILogger<Phi3Controller> logger)
    {
        _configuration = configuration;
        _logger = logger;

        _defaultSystemPrompt = configuration["DefaultSystemPrompt"];
        _model = new Model(configuration["ModelPath"]);
        _tokenizer = new Tokenizer(_model);
        _tokenizerStream = _tokenizer.CreateStream();
    }

    [HttpPost("generate-response")]
    public async Task GenerateResponse([FromBody] ChatRequest request)
    {
        var requestSystemPrompt = request.Messages.FirstOrDefault(p => p.Role == "system")?.Content;
        var systemPrompt = requestSystemPrompt ?? _defaultSystemPrompt;
        var userPropmpt = request.Messages.FirstOrDefault(p => p.Role == "user")?.Content;

        if (requestSystemPrompt != null)
        {
            _logger.LogInformation($"Requested System Prompt: {requestSystemPrompt}");
        }

        _logger.LogInformation($"User Prompt: {userPropmpt}");

        var fullPrompt = $"<|system|>{systemPrompt}<|end|>" +
                         $"<|user|>{userPropmpt}<|end|>" +
                         $"<|assistant|>";

        Response.ContentType = "text/plain";
        await foreach (var token in GenerateAiResponse(fullPrompt))
        {
            if (string.IsNullOrEmpty(token))
            {
                break;
            }
            await Response.WriteAsync(token);
            await Response.Body.FlushAsync();
        }
    }

    private async IAsyncEnumerable<string> GenerateAiResponse(string fullPrompt)
    {
        var tokens = _tokenizer.Encode(fullPrompt);

        // Parse configuration values once
        if (!int.TryParse(_configuration["GeneratorParams:max_length"], out var maxLength) ||
            !bool.TryParse(_configuration["GeneratorParams:past_present_share_buffer"], out var pastPresentShareBuffer) ||
            !int.TryParse(_configuration["GeneratorParams:num_return_sequences"], out var numReturnSequences) ||
            !float.TryParse(_configuration["GeneratorParams:temperature"], out var temperature) ||
            !int.TryParse(_configuration["GeneratorParams:top_k"], out var topK) ||
            !float.TryParse(_configuration["GeneratorParams:top_p"], out var topP))
        {
            throw new InvalidOperationException("Invalid configuration settings.");
        }

        var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", maxLength);
        generatorParams.SetSearchOption("past_present_share_buffer", pastPresentShareBuffer);
        generatorParams.SetSearchOption("num_return_sequences", numReturnSequences);
        generatorParams.SetSearchOption("temperature", temperature);
        generatorParams.SetSearchOption("top_k", topK);
        generatorParams.SetSearchOption("top_p", topP);

        generatorParams.SetInputSequences(tokens);

        var generator = new Generator(_model, generatorParams);

        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();

            var output = GetOutputTokens(generator);
            if (string.IsNullOrEmpty(output))
            {
                yield break;
            }

            yield return output;
        }

        await Task.CompletedTask;
    }

    private string GetOutputTokens(Generator generator)
    {
        var outputTokens = generator.GetSequence(0);
        var newToken = outputTokens[^1];
        var token = _tokenizerStream.Decode(newToken);

        return token;
    }
}