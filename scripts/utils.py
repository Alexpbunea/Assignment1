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
    print(f"File saved in: {output_path}")


#necessary to answer the questions in the correct format
def yes_or_no(answer):
    if answer.lower() in ("yes", "y"):
        return True
    elif answer.lower() in ("no", "n"):
        return False
    else:
        print("Please answer with 'yes', 'y', 'n' or 'no'.")
        return yes_or_no(input())



def extract_source_tables(data, db_schemas): #extracts the tables directly from the MYSQL files.
    
    lista = []
    dicionario = {}

    db_id2 = ""  

    for entry in data:
        db_id = entry.get("db_id")
        sql_query = entry.get("SQL", "")
        question_text = entry.get("question", "").lower()

        
        tables = re.findall(r'FROM\s+([`"\w.]+)', sql_query, re.IGNORECASE)
        tables += re.findall(r'JOIN\s+([`"\w.]+)', sql_query, re.IGNORECASE)
        cleaned_tables = set(table.strip('`"') for table in tables)  

        if db_id in db_schemas:
            for table in db_schemas[db_id]:  
                if table in question_text:
                    cleaned_tables.add(table)  
        
        cleaned_tables = list(cleaned_tables)

        if db_id != db_id2:
            if db_id2 != "":
                dicionario[db_id2] = lista
            lista = []  

        lista.append(cleaned_tables)
        db_id2 = db_id  

    if db_id2 != "":
        dicionario[db_id2] = lista

    return dicionario




def merge_datasets(m_schema, questions_sql):
    source_tables = extract_source_tables(questions_sql, m_schema)    
    schema_dict = {entry["db_id"]: entry["tables"] for entry in m_schema}
    
    merged = {}
    indices = {}

    for qa in questions_sql:
        db_id = qa.get("db_id")
            
        if db_id not in indices:
            indices[db_id] = 0
        
        n = indices[db_id]
        a = source_tables.get(db_id)[n]
        indices[db_id] += 1

        if db_id not in merged:
            merged[db_id] = {
                "db_id": db_id,
                "tables": schema_dict.get(db_id, []),  
                "qa": []  
            }        
        merged[db_id]["qa"].append({
            "question": qa.get("question"),
            "output": a  
        })

    return list(merged.values())





def create_structure_for_jsonl_file(data):
    """
    Genera a JOSNL file with this structure
    {
      "input": "Database ID: ...\nSchema: ...\nQuestion: ...",
      "output": "..." #the output tables
    }

    """
    json_lines = [] 
    
    for entry in data:
        db_id = entry["db_id"]
        schema_str = entry["tables"] 
        
        
        for qa_pair in entry["qa"]:
            question = qa_pair["question"]
            output_text = qa_pair["output"]
            
            prompt = f"Database ID: {db_id}\nQuestion: {question}\nSelect table(s) from: {schema_str}"
            json_line = {"input": prompt, "output": output_text}
            json_lines.append(json_line)
    
    return json_lines
