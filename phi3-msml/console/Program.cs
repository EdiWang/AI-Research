using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.Phi;
using Microsoft.ML.GenAI.Phi.Extension;
using Microsoft.ML.Tokenizers;
using Microsoft.SemanticKernel;
using static TorchSharp.torch;
using TorchSharp;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Phi3MicrosoftMLConsole;

internal class Program
{
    static async Task Main(string[] args)
    {
        var device = "cuda";
        if (device == "cuda")
        {
            InitializeDeviceType(DeviceType.CUDA);
        }

        var defaultType = ScalarType.Float16;
        manual_seed(1);
        set_default_dtype(defaultType);
        var weightFolder = @"E:\ai-models\Phi-3-mini-4k-instruct";
        var tokenizerPath = Path.Combine(weightFolder, "tokenizer.model");
        var tokenizer = Phi3TokenizerHelper.FromPretrained(tokenizerPath);
        var model = Phi3ForCasualLM.FromPretrained(weightFolder, "config.json", layersOnTargetDevice: -1, quantizeToInt8: true);
        var pipeline = new CausalLMPipeline<LlamaTokenizer, Phi3ForCasualLM>(tokenizer, model, device);

        var kernel = Kernel.CreateBuilder()
            .AddGenAIChatCompletion(pipeline)
            .Build();

        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage("you are a helpful assistant");
        chatHistory.AddUserMessage("What's the capital city of USA?");

        await foreach (var response in chatService.GetStreamingChatMessageContentsAsync(chatHistory))
        {
            Console.Write(response);
        }
    }
}