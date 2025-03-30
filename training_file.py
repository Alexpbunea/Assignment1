from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json
import torch


def training_model(input_data = "./jsons/train_dataset.jsonl"):
    model_name = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        if device == "cuda":
            torch.clear_autocast_cache() #cleaning the vram just in case
    except Exception as e:
        print("[ERROR]: Not able to clean the vram of the GPU")

    # Ruta a tu dataset JSONL
    dataset_path = input_data
    train_data = Dataset.from_json(dataset_path)

    # Función de tokenización
    def tokenize_function(batch):
        # Tokenizar el campo "input"
        inputs = batch["input"]
        inputs = [(" ".join(i) if isinstance(i, list) else i).strip() for i in inputs]
        inputs = [i if i else "No input provided." for i in inputs]

        # Tokenizar el campo "output"
        outputs = batch["output"]
        outputs = [(", ".join(str(o) for o in i) if isinstance(i, list) else i).strip() for i in outputs]
        outputs = [o if o else "No output provided." for o in outputs]

        
        input_encodings = tokenizer(inputs, truncation=True, max_length=512, padding="max_length")
        output_encodings = tokenizer(outputs, truncation=True, max_length=64, padding="max_length")

        
        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in output]
            for output in output_encodings["input_ids"]
        ]

        
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": labels,
        }

    # Tokenizar el dataset
    tokenized_train = train_data.map(tokenize_function, batched=True, num_proc=8)

    # Dividir el dataset en entrenamiento y validación (80% / 20%)
    train_data_split = tokenized_train.train_test_split(test_size=0.2)
    train_dataset = train_data_split["train"]
    eval_dataset = train_data_split["test"]

    # Data collator para secuencias
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results",
        eval_steps=500,  
        learning_rate=4e-5,
        per_device_train_batch_size=4, #recuerda a probar con 8 ya que es posible que el overfitting sea por el 4
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        report_to=[],  # Desactiva wandb
        logging_steps=100,
        gradient_accumulation_steps=4,  # O más si tu GPU lo permite
        lr_scheduler_type="polynomial"
    )

    # Inicializar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Proporcionar el conjunto de validación
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar el modelo fine-tuneado
    model.save_pretrained("./fine_tuned_t5_small")
    tokenizer.save_pretrained("./fine_tuned_t5_small")
