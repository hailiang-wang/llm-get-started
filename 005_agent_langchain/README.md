# Agent built with LangChain, and Chatopera Cloud

[By themselves, language models can't take actions - they just output text. A big use case for LangChain is creating agents. Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish. This is often achieved via tool-calling.](https://python.langchain.com/docs/tutorials/agents/)

## RAG & Agent

* https://gaodalie.substack.com/p/langchain-mcp-rag-ollama-the-key
* https://python.langchain.com/docs/tutorials/rag/
* https://python.langchain.com/docs/tutorials/qa_chat_history/
* https://python.langchain.com/docs/tutorials/agents/


## Start

First, install ollama, next, run `ollama pull mistral-nemo:latest`, checkout [mistral-nemo](https://mistral.ai/news/mistral-nemo).


Next, install pip deps.

```
pip install -r requirements.txt
```

Then, config env file.

```
cp sample.env .env # Modify key-values in .env
```

At last, run
```
./start.sh
```

![alt text](../assets/media/1748600958462.png)