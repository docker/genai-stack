# GenAI Stack
This GenAI application stack will get you started building your
own GenAI application in no time.  
The demo applications can serve as inspiration or as a starting point.

# Configure

Create a `.env` file from the environment template file `env.example`

## LLM Configuration
MacOS and Linux users can use any LLM that's available via Ollama. Check the "tags" section under the model page you want to use on https://ollama.ai/library and write the tag for the value of the environment variable `LLM=` in th e`.env` file.  
All platforms can use GPT-3.5-turbo and GPT-4 (bring your own API keys for OpenAIs models).

**MacOS**  
Install [Ollama](https://ollama.ai) in MacOS and start it before running `docker compose up`.  

**Linux**  
No need to install Ollama manually, it will run in a container as 
part of the stack when running with the Linus profile: `run docker compose up --profile linux`.  

**Windows**  
Not supported by Ollama, so Windows users need to generate a OpenAI API key and configure the stack to use `gpt-3.5` or `gpt-4` in the `.env` file.
# Develop
**To start everything**
```
docker compose up
```
If changes to build scripts has been made, **rebuild**
```
docker compose up --build
```

To enter **watch mode** (auto rebuild on file changes).  
First start everything, then in new terminal:
```
docker compose alpha watch
```

**Shutdown**  
Is health check fails or containers doesn't start up as expected, shutdown
completely to start up again.
```
docker compose down
```

# Applications
## App 1 - Support Agent Bot  

UI: http://localhost:8501  
DB client: http://localhost:7474

- answer support question based on recent entries
- provide summarized answers with sources
- demonstrate difference between 
    - RAG Disabled (pure LLM reponse)
    - RAG Enabled (vector + knowledge graph context)
- allow to generate a high quality support ticket for the current conversation based on the style of highly rated questions in the database.

![](.github/media/app1-rag-selector.png)  
*(Chat input + RAG mode selector)*

![](.github/media/app1-generate.png)  
*(CTA to auto generate support ticket draft)*

![](.github/media/app1-ticket.png)  
*(UI of the auto generated support ticket draft)*

--- 

##  App 2 Loader

UI: http://localhost:8502  
DB client: http://localhost:7474

- import recent SO data for certain tags into a KG
- embed questions and answers and store in vector index
- UI: choose tags, run import, see progress, some stats of data in the database
- Load high ranked questions (regardless of tags) to support the ticket generation feature of App 1.

![](.github/media/app2-ui-1.png)  
![](.github/media/app2-model.png)
