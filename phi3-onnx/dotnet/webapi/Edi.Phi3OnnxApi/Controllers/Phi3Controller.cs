using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Edi.Phi3OnnxApi.Controllers;

[ApiController]
[Route("[controller]")]
public class Phi3Controller : Controller
{
    private static Model _model = null;
    private readonly Tokenizer _tokenizer = null;
    private static readonly string DefaultSystemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

    public Phi3Controller()
    {
        var modelPath = @"E:\Workspace\phi-35-test\model\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4";
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
    }

    [HttpPost("generate-response")]
    public async Task GenerateResponse([FromBody] string userPrompt)
    {
        var fullPrompt = $"<|system|>{DefaultSystemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";

        Response.ContentType = "text/plain";
        await foreach (var token in GenerateAiResponse(fullPrompt))
        {
            if (string.IsNullOrEmpty(token))
            {
                break;
            }
            await Response.WriteAsync(token);
            await Response.Body.FlushAsync(); // Flush the response stream to send the token immediately
        }
    }

    private async IAsyncEnumerable<string> GenerateAiResponse(string fullPrompt)
    {
        var tokens = _tokenizer.Encode(fullPrompt);

        var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", 2048);
        generatorParams.SetSearchOption("past_present_share_buffer", false);
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
            yield return output; // Yield each token as it's generated
        }
    }

    private string GetOutputTokens(Generator generator, Tokenizer tokenizer)
    {
        var outputTokens = generator.GetSequence(0);
        var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
        return tokenizer.Decode(newToken);
    }
}