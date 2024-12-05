using System.ComponentModel.DataAnnotations;

namespace Edi.Phi3OnnxApi.Models;

public class ChatCompletionRequest
{
    [Required]
    public Message[] Messages { get; set; }
}

public class Message
{
    public string Role { get; set; } = "user";

    [Required]
    public string Content { get; set; }
}