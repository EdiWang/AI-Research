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
    private static string _defaultSystemPrompt;

    public Phi3Controller(IConfiguration configuration, ILogger<Phi3Controller> logger)
    {
        _configuration = configuration;
        _logger = logger;

        _defaultSystemPrompt = configuration["DefaultSystemPrompt"];
        _model = new Model(configuration["ModelPath"]);
        _tokenizer = new Tokenizer(_model);
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

        var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", int.Parse(_configuration["GeneratorParams:max_length"]!));
        generatorParams.SetSearchOption("past_present_share_buffer", bool.Parse(_configuration["GeneratorParams:past_present_share_buffer"]!));
        generatorParams.SetInputSequences(tokens);

        var generator = new Generator(_model, generatorParams);

        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();

            var output = GetOutputTokens(generator, _tokenizer);
            if (string.IsNullOrEmpty(output))
            {
                break;
            }

            yield return output;
        }

        await Task.CompletedTask;
    }

    private string GetOutputTokens(Generator generator, Tokenizer tokenizer)
    {
        var outputTokens = generator.GetSequence(0);
        var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);

        // Workaround bug where every response ends with invalid character
        if (newToken[0] == 32007) return string.Empty;

        var token = tokenizer.Decode(newToken);
        return token;
    }
}