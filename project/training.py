import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

class ModelTrainer:
    """
    Train models for bias detection and debiasing.
    """

    def __init__(self):
        pass

    def _load_dataset(self, file_path, required_columns, drop_na_subset, test_size=0.2, stratify_col=None, preprocess_fn=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        
        # Check columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")

        df = df.dropna(subset=drop_na_subset)
        
        if preprocess_fn:
            df = preprocess_fn(df)

        print(f"Records after cleaning: {len(df)}")

        stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=stratify, 
            random_state=42
        )
        
        return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

    def train_bias_detection(self, 
                             model_name="dccuchile/bert-base-spanish-wwm-cased", 
                             dataset_path="dataset-bias-identification.csv",
                             output_dir="model-bias-detection"):
        """
        Train a text classification model to detect biases.
        """
        print(f"Training Bias Detection Model: {model_name}")
        
        def preprocess_bias_df(df):
            df['label'] = df['label_id'].astype(int)
            return df

        dataset_train, dataset_test = self._load_dataset(
            dataset_path, 
            required_columns=['texto', 'label_id'], 
            drop_na_subset=['label_id'],
            stratify_col='label_id',
            preprocess_fn=preprocess_bias_df
        )

        # Label mapping
        id2label = {0: 'sesgo_genero_masculino', 1: 'sesgo_genero_femenino', 2: 'sesgo_edad_joven', 3: 'sesgo_edad_mayor', 4: 'sesgo_nacionalidad', 5: 'neutral_inclusivo'}
        label2id = {v: k for k, v in id2label.items()}

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["texto"], padding="max_length", truncation=True, max_length=128)

        encoded_dataset_train = dataset_train.map(tokenize_function, batched=True)
        encoded_dataset_test = dataset_test.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=f"./results-{output_dir}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_dir=f'./logs-{output_dir}',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset_train,
            eval_dataset=encoded_dataset_test,
            tokenizer=tokenizer,
        )

        trainer.train()
        print(trainer.evaluate())

        model.save_pretrained(f"./{output_dir}")
        tokenizer.save_pretrained(f"./{output_dir}")
        print(f"Bias detection model saved to {output_dir}")

    def train_debiasing(self, 
                        model_name="google/mt5-base", 
                        dataset_path="dataset-debiasing.csv",
                        output_dir="model-debiasing"):
        """
        Train a seq2seq model for debiasing.
        """
        print(f"Training Debiasing Model: {model_name}")

        dataset_train, dataset_test = self._load_dataset(
            dataset_path,
            required_columns=['texto_sesgado', 'texto_corregido'],
            drop_na_subset=['texto_sesgado', 'texto_corregido']
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def preprocess_function(examples):
            inputs = ["debias: " + text for text in examples["texto_sesgado"]]
            targets = examples["texto_corregido"]
            
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
            labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        encoded_dataset_train = dataset_train.map(preprocess_function, batched=True)
        encoded_dataset_test = dataset_test.map(preprocess_function, batched=True)

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./results-{output_dir}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            predict_with_generate=True,
            load_best_model_at_end=True,
            logging_dir=f'./logs-{output_dir}',
            save_total_limit=2,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset_train,
            eval_dataset=encoded_dataset_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        print(trainer.evaluate())

        model.save_pretrained(f"./{output_dir}")
        tokenizer.save_pretrained(f"./{output_dir}")
        print(f"Debiasing model saved to {output_dir}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Uncomment to train bias detection
    # trainer.train_bias_detection()
    
    # Uncomment to train debiasing
    # trainer.train_debiasing()
