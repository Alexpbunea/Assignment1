import json
import re
from utils import *

def extract_source_tables(train_json_path):
    """
    Procesa un archivo JSON de entrenamiento y extrae las tablas fuente mencionadas en las consultas SQL.

    Args:
        train_json_path (str): Ruta al archivo JSON de entrenamiento.

    Returns:
        dict: Diccionario con el db_id como clave y las tablas fuente como valores.
    """

    data = load_json(train_json_path) #calling the function from utils.py

    source_tables = {}

    for entry in data:
        db_id = entry.get("db_id")
        sql_query = entry.get("SQL", "")

        # Extraer nombres de tablas usando una expresión regular
        tables = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        tables += re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE)

        # Eliminar duplicados y almacenar en el diccionario
        source_tables[db_id] = list(set(tables))

    return source_tables



train_json_path = "./dev/dev.json"  # Cambia esta ruta al archivo JSON real
source_tables = extract_source_tables(train_json_path)

# Imprimir las tablas fuente extraídas
for db_id, tables in source_tables.items():
    print(f"Database ID: {db_id}")
    print(f"Source Tables: {tables}")
    print("-" * 40)