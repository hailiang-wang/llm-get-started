# https://mer.vin/2025/01/ollama-reasoning-chatbot-code/
import ollama
import gradio as gr

def chat_with_ollama(message, history):
    # Initialize empty string for streaming response
    response = ""
    
    # Convert history to messages format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    # Add history messages
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:  # Only add assistant message if it exists
            messages.append({"role": "assistant", "content": h[1]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    completion = ollama.chat(
        model="deepseek-r1:14b",
        messages=messages,
        stream=True  # Enable streaming
    )
    
    # Stream the response
    for chunk in completion:
        if 'message' in chunk and 'content' in chunk['message']:
            content = chunk['message']['content']
            # Handle <think> and </think> tags
            content = content.replace("<think>", "Thinking...").replace("</think>", "\n\n Answer:")
            response += content
            yield response

# Create Gradio interface with Chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter your message here...")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        history[-1][1] = ""
        for chunk in chat_with_ollama(history[-1][0], history[:-1]):
            history[-1][1] = chunk
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":  
    demo.launch()