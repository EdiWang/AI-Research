﻿using Microsoft.ML.OnnxRuntimeGenAI;

namespace Phi3OnnxConsole;

internal class Program
{
    static void Main(string[] args)
    {
        var modelPath = @"E:\Workspace\phi-35-test\model\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4";
        var model = new Model(modelPath);
        var tokenizer = new Tokenizer(model);
        var tokenizerStream = tokenizer.CreateStream();

        var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

        // chat start
        Console.WriteLine(@"Ask your question. Type an empty string to Exit.");


        // chat loop
        while (true)
        {
            // Get user question
            Console.WriteLine();
            Console.Write(@"Q: ");
            var userQ = Console.ReadLine();
            if (string.IsNullOrEmpty(userQ))
            {
                break;
            }

            // show phi3 response
            Console.Write("Phi3: ");
            var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
            var tokens = tokenizer.Encode(fullPrompt);

            var generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetSearchOption("past_present_share_buffer", false);
            generatorParams.SetInputSequences(tokens);

            var generator = new Generator(model, generatorParams);
            while (!generator.IsDone())
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();
                var outputTokens = generator.GetSequence(0);

                var newToken = outputTokens[^1];
                var output = tokenizerStream.Decode(newToken);

                //var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);

                //// Workaround bug where every response ends with invalid character
                //if (newToken[0] == 32007) continue;

                //var output = tokenizer.Decode(newToken);
                Console.Write(output);
            }
            Console.WriteLine();
        }
    }
}