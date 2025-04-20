# ollama-get-started
Get started with Ollama for LLMs.

Install by installer - [https://ollama.com/download](https://ollama.com/download).

Next, clone codes.

```
git clone https://github.com/hailiang-wang/ollama-get-started.git
cd ollama-get-started
```

## DeepSeek R1

Video - https://www.youtube.com/watch?v=pLWzBA7mmgE
Model - https://ollama.com/library/deepseek-r1


```
# download deepseek model and launch
ollama pull deepseek-r1
ollama run deepseek-r1
ollama ps # check running status

cd deepseek
pip install -r requirement.txt
python app.py
# Run Gradio to chat in Browser.
```

### Chat via Console

![alt text](assets/media/1745139699936.png)

### Chat via Browser

![alt text](assets/media/1745139758807.png)

## FAQs

* [ollama FAQs](https://github.com/ollama/ollama/blob/main/docs/faq.md)
* [ollama Community](https://discord.com/channels/1128867683291627614/1211804431340019753)