# GenAI Stack
This GenAI application stack will get you started building your
own GenAI application in no time.  
The demo applications can serve as inspiration or as a starting point.

# Configuration

Create a `.env` file from the environment template file `env.example`

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
