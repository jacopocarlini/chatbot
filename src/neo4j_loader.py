from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "verysecret"))

def insert_triple(subj, rel, obj):
    with driver.session() as session:
        session.execute_write(_insert, subj, rel, obj)

def _insert(tx, subj, rel, obj):
    tx.run("""
    MERGE (a:Entity {name: $subj})
    MERGE (b:Entity {name: $obj})
    MERGE (a)-[:RELATION {type: $rel}]->(b)
    """, subj=subj, rel=rel, obj=obj)
