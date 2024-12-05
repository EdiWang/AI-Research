namespace Edi.Phi3OnnxApi.Models;

public class ChatCompletionRequest
{
    public Message[] Messages { get; set; }
}

public class Message
{
    public string Role { get; set; }
    public string Content { get; set; }
}