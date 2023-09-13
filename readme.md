# LangChain Docker Starter Kits


# Configuration

Use an `.env` file for the OPENAI_API_KEY (see `env.example`)

* Optionally Neo4j credentials for Aura or remote database (TODO) from env file

* configure initial page loads?
* configure Q&A prompt via env-variable?

## App 1 - KG + Embedding Construction from Wikipedia

* dataset: countries of the world
    * pre-seeded database
* load/add (additional) wikipedia pages
    * provide examples history / politics / EU / …
* create embeddings
* create knowledge graph
* UI: Neo4j workspace (centrally hosted or docker image)

### Endpoints

* `/add_page?title=pagename`
* `/clear`

##  App 2 - Q&A Chatbot - Python Dev
ask questions about the dataset form App 1
use RAG for answering
vector index and/or cypher-gen CypherQAChain?
Streamlit App Python
? store - questions + answers + embedding + upvotes ? (conversational memory)

### Endpoints

* `/answer?title=

For control plane / extension - or env-variables

* `/prompt` POST / GET
* `/configure` (temperature, top-k, ...)

##  App 3 - Report Generator

* Generate Reports based on user question
    * eg.. charts of the GDP of European countries
    * or PDF of the history of African Independence
    * or 5 slides presentation about Mexico
* could use additional containers for the rendering (e.g. markdown to pdf or chart-generator?)
