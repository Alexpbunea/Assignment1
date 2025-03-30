import json
import difflib
import re
import ast
from utils import *

file_path = "./generated_outputs(2).jsonl"


def check_similarity(item, candidate_lower):
    """
    Compara el output_value con los candidatos y devuelve True si hay coincidencia
    o similitud suficiente, de lo contrario devuelve False.
    """

    # Buscar entre los candidatos aquel que tenga similitud >= 80%
    mejor_ratio = 0
    mejor_candidato = None
    for candidate in candidate_lower:
        ratio = difflib.SequenceMatcher(None, item, candidate).ratio()
        if ratio >= 0.5 and ratio > mejor_ratio:
            mejor_ratio = ratio
            mejor_candidato = candidate

    return mejor_candidato if not None else False



def refinining(raw_generated_outputs_path, refined_generated_outputs_path):
    with open(raw_generated_outputs_path, 'r', encoding='utf-8') as f:
        # Leer cada línea y cargarla como un diccionario
        data = [json.loads(line) for line in f]

    final_jsonl = []

    for i in range(len(data)):
        registro = data[i]
        output_value = registro.get('output', '').lower().split(", ")
        output_value = set(output_value) #delete duplicates

        input_text = registro.get('input', '')

        #Looks for the original dataset tables in the input
        match = re.search(r"Select table\(s\) from:\s*(\[.*\])", input_text)
        if match:
            table_list_str = match.group(1)
            try:
                candidate_list = ast.literal_eval(table_list_str)
            except Exception as e:
                print(f"[Error]: When evaluating the list: {e}")
                candidate_list = []
        else:
            candidate_list = []

        
        candidate_lower = [str(elem).lower() for elem in candidate_list]
        #print(candidate_lower, " ---------- ", output_value)
        
        
        new_output_value = []
        for item in output_value:
            if item not in candidate_lower:
                new_candidate = check_similarity(item, candidate_lower)
                if new_candidate: 
                    new_output_value.append(new_candidate)
            else:
                new_output_value.append(item)  # Si ya está en candidate_lower, conservarlo
        
        registro['output'] = new_output_value
        final_jsonl.append(registro)

    
    with open(refined_generated_outputs_path, "w", encoding="utf-8") as fout:
        for registro in final_jsonl:
            fout.write(json.dumps(registro) + "\n")

