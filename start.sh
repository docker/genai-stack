# docker run -d --name neo4j -e NEO4J_AUTH=neo4j/password -p 7687:7687 -p 7474:7474 neo4j:latest
# docker ps -qaf name=neo4j

docker compose up --build --force-recreate
