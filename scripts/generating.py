from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import torch


def generate(input_data, model_path="./fine_tuned_t5_small", output_path="./generated_outputs.jsonl"):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    
    inputs = [example["input"] for example in input_data]  # EXTRACT ONLY THE INPUTS

    
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    
    decoded_outputs = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        
        inputs_encodings = tokenizer(batch_inputs, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        
        
        inputs_encodings = {key: val.to(device) for key, val in inputs_encodings.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs_encodings["input_ids"],
                attention_mask=inputs_encodings["attention_mask"],
                max_length=64,  
                num_beams=10,
                early_stopping=True
            )

        batch_decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
 
        decoded_outputs.extend(batch_decoded_outputs)

    with open(output_path, "w", encoding="utf-8") as f:
        for input_example, output in zip(input_data, decoded_outputs):
            f.write(json.dumps({"input": input_example["input"], "output": output}) + "\n")

