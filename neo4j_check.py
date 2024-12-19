from neo4j import GraphDatabase

uri = "bolt://10.11.28.104/:7687"  # or your specific Neo4j URI
username = "neo4j"  # default username
password = "@neolenovoserver"  # change if necessary

driver = GraphDatabase.driver(uri, auth=(username, password))

def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        for record in result:
            print(record["message"])

test_connection()
