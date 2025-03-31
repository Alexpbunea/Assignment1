import json
from utils import *

#file_path = "./final_output.jsonl" #output of the model
#file_path2 = "./jsons/dev_dataset.jsonl" #groundtruth

"""
When evaluating I treat several cases to evaluate good results: (model == groundtruth)
- Perfect example: ["schools", "satscores"] == ["schools", "satscores"]. Acurracy_sum += 1
- Almost perfect: ["schools"] == ["schools", "satscores"]. In this case, acurracy_sum += 0.5 and false_negative += 0.5
- Almost perfect 2: ["schools", "satscores", "rpfm"] == ["schools", "satscores"]. In this case, acurracy_sum += 0.66 and false_positive += 0.33
"""


def evaluate(file_path, file_path2):
    a = load_json(file_path2)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    accuracy_sum = 0
    perfect_examples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    b = len(a)

    for i in range(b):
        output = a[i].get('output')
        c = data[i].get('output')
        
        output_lower = [elem.lower() for elem in output]
        c_lower = [elem.lower() for elem in c]
        
        if set(c_lower) == set(output_lower):
            accuracy_sum += 1
            perfect_examples += 1
            continue
        else:
            for elem in c_lower:
                if elem in output_lower:
                    #c = results of the model, output_lower = results of the database
                    #print(c, " -------------- ", output, " ------------ > ", elem)
                    if len(output_lower) > len(c_lower):
                        accuracy_sum += 1/len(output_lower)
                        true_positives += 1/len(output_lower)
                    else:
                        accuracy_sum += 1/len(c_lower)
                        true_positives += 1/len(c_lower)
                    continue
                else:
                    if len(output_lower) > len(c_lower):
                        false_negatives += 1/len(output_lower)
                    else:
                        false_positives += 1/len(c_lower)

    #print("Suma precisión acumulada:", accuracy_sum)
    #print("Número total de muestras:", b)
    #print("Contador perfectos:", perfect_examples)
    #print("Suma parcial recall:", true_positives)

    accuracy = accuracy_sum / b
    precision = (perfect_examples + true_positives) / (perfect_examples + true_positives + false_positives) if (perfect_examples + true_positives + false_positives) > 0 else 0
    recall = (perfect_examples + true_positives) / (perfect_examples + true_positives + false_negatives) if (perfect_examples + true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
