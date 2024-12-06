using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace Phi3SkOnnxConsole;

internal class Program
{
    static async Task Main(string[] args)
    {
        var modelPath = @"E:\Workspace\phi-35-test\model\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4";

        // create kernel
        var builder = Kernel.CreateBuilder();
        builder.AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
        var kernel = builder.Build();

        var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

        // create chat
        var chat = kernel.GetRequiredService<IChatCompletionService>();
        var history = new ChatHistory();

        // run chat
        while (true)
        {
            Console.Write("Q: ");
            var userQ = Console.ReadLine();
            if (string.IsNullOrEmpty(userQ))
            {
                break;
            }

            history.AddSystemMessage(systemPrompt);
            history.AddUserMessage(userQ);

            Console.Write($"Phi3: ");
            var response = "";
            var result = chat.GetStreamingChatMessageContentsAsync(history);
            await foreach (var message in result)
            {
                Console.Write(message.Content);
                response += message.Content;
            }
            history.AddAssistantMessage(response);
            Console.WriteLine(Environment.NewLine);

            var key = Console.ReadKey();
            if (key.Key == ConsoleKey.Escape)
            {
                break;
            }
        }
    }
}