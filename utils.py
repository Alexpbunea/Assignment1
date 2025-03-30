import json
import os
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def write_json(directory, file_name, data):
    output_path = os.path.join(directory, file_name)

    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=1, ensure_ascii=False)

    #json.dumps(data, indent=1, ensure_ascii=False)
    print(f"File saved in: {output_path}")

def yes_or_no(answer):
    if answer.lower() in ("yes", "y"):
        return True
    elif answer.lower() in ("no", "n"):
        return False
    else:
        print("Please answer with 'yes', 'y', 'n' or 'no'.")
        return yes_or_no(input())



def extract_source_tables(data, db_schemas):
    """
    Extrae las tablas fuente mencionadas en las consultas SQL y analiza la pregunta en lenguaje natural
    para incluir tablas adicionales relevantes.

    Args:
        data (list): Lista de entradas JSON con claves "db_id", "SQL" y "question".
        db_schemas (dict): Diccionario con db_id como clave y lista de tablas como valor.

    Returns:
        dict: Diccionario con db_id como clave y listas de tablas como valores.
    """
    source_tables = {}
    lista = []
    dicionario = {}

    db_id2 = ""  # Para rastrear el último db_id procesado

    for entry in data:
        db_id = entry.get("db_id")
        sql_query = entry.get("SQL", "")
        question_text = entry.get("question", "").lower()

        # Extraer nombres de tablas desde la consulta SQL
        tables = re.findall(r'FROM\s+([`"\w.]+)', sql_query, re.IGNORECASE)
        tables += re.findall(r'JOIN\s+([`"\w.]+)', sql_query, re.IGNORECASE)

        # Limpiar nombres de tablas (quitar comillas o caracteres innecesarios)
        cleaned_tables = set(table.strip('`"') for table in tables)  # ✅ Usamos `set()` para evitar duplicados

        # 🔹 Identificar palabras clave en la pregunta y mapearlas a tablas 🔹
        if db_id in db_schemas:
            for table in db_schemas[db_id]:  
                if table in question_text:
                    cleaned_tables.add(table)  # ✅ Se usa `set.add()` en vez de `.append()`

        # Convertimos el set de nuevo a lista para mantener la estructura original
        cleaned_tables = list(cleaned_tables)

        # Si estamos en un nuevo db_id, guardamos las tablas anteriores
        if db_id != db_id2:
            if db_id2 != "":
                dicionario[db_id2] = lista
            lista = []  # Reiniciar lista para el nuevo db_id

        # Agregar las tablas extraídas a la lista correspondiente a este db_id
        lista.append(cleaned_tables)
        db_id2 = db_id  

    # Guardar las tablas del último db_id procesado
    if db_id2 != "":
        dicionario[db_id2] = lista

    return dicionario




def merge_datasets(m_schema, questions_sql):
    # Obtener las tablas fuente por db_id
    source_tables = extract_source_tables(questions_sql, m_schema)
    
    # Creamos un diccionario para acceder al M-Schema por db_id
    schema_dict = {entry["db_id"]: entry["tables"] for entry in m_schema}
    
    # Diccionario para agrupar resultados por db_id
    merged = {}
    # Diccionario para llevar el control del índice (posición) para cada db_id
    indices = {}

    for qa in questions_sql:
        db_id = qa.get("db_id")
        
        # Inicializamos el índice para este db_id si no existe
        if db_id not in indices:
            indices[db_id] = 0
        
        # Obtenemos el output correspondiente según el índice para este db_id
        n = indices[db_id]
        a = source_tables.get(db_id)[n]
        indices[db_id] += 1

        # Si el db_id no ha sido agregado, lo creamos con una lista de pares pregunta/output
        if db_id not in merged:
            merged[db_id] = {
                "db_id": db_id,
                "tables": schema_dict.get(db_id, []),  # Aseguramos que sea una lista
                "qa": []  # Lista que contendrá diccionarios con pregunta y output
            }
        
        # Agregamos la pregunta y su output de forma independiente
        merged[db_id]["qa"].append({
            "question": qa.get("question"),
            "output": a  # Solo se asigna la lista (o elemento) en la posición n
        })

    # Convertir el diccionario a lista y retornar
    return list(merged.values())





def create_structure_for_jsonl_file(data):
    """
    Genera un conjunto de líneas JSON donde cada línea tiene la estructura:
    {
      "input": "Database ID: ...\nSchema: ...\nQuestion: ...",
      "output": "source_tables: ..."
    }
    Retorna una lista de líneas JSON (en lugar de escribirlas directamente en un archivo).
    """
    json_lines = []  # Lista para almacenar las líneas JSON
    
    for entry in data:
        db_id = entry["db_id"]
        schema_str = entry["tables"]  # Ya es una lista, la podemos usar directamente
        
        # Ahora iteramos sobre la lista de preguntas con sus respectivos outputs
        for qa_pair in entry["qa"]:
            question = qa_pair["question"]
            output_text = qa_pair["output"]
            
            prompt = f"Database ID: {db_id}\nQuestion: {question}\nSelect table(s) from: {schema_str}"
            json_line = {"input": prompt, "output": output_text}
            json_lines.append(json_line)
    
    return json_lines
