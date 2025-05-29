# LLM Get Started
Get started with LLMs.

First, Install 

* Python 3.11
* Pytorch 2.6+
* ollama by installer - [https://ollama.com/download](https://ollama.com/download).

Next, clone codes.

```
git clone https://github.com/hailiang-wang/llm-get-started.git
cd llm-get-started
```

## 001_transformer

Transformer basics, original algorithm fundamentals for Transformer Model.

[README](./001_transformer/README.md)


## 002_deepseek

Run inferences and reasoning with DeepSeek R1.

Video - https://www.youtube.com/watch?v=pLWzBA7mmgE
Model - https://ollama.com/library/deepseek-r1
Code - https://mer.vin/2025/01/ollama-reasoning-chatbot-code/

```
# download deepseek model and launch
ollama pull deepseek-r1:14b
ollama run deepseek-r1:14b
ollama ps # check running status

cd 002_deepseek
pip install -r requirement.txt
python app.py
# Run Gradio to chat in Browser.
```

### Chat via Console

![alt text](assets/media/1745139699936.png)

### Chat via Browser

![alt text](assets/media/1745139758807.png)


## 003_langchain

Langchain for RAG

Based on [blog](https://dev.to/ajmal_hasan/setting-up-ollama-running-deepseek-r1-locally-for-a-powerful-rag-system-4pd4) and [code](https://github.com/Ajmal0197/DeepseekOllamaRag/blob/main/app.py).

First, make sure works `DeepSeek R1` above are done, because Langchain RAG App below depends on `DeepSeek R1`.

```
cd 003_langchain
pip install -r requirements.txt
streamlit run app.py
```

![alt text](assets/media/1745142574757.png)

Trace log - 

![alt text](assets/media/1745142610493.png)

## FAQs

* [ollama FAQs](https://github.com/ollama/ollama/blob/main/docs/faq.md)
* [ollama Community](https://discord.com/channels/1128867683291627614/1211804431340019753)