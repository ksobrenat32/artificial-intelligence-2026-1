import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
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
        self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

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

    def generate_embeddings_vector_store(self, 
                                         dataset_path="dataset-debiasing.csv",
                                         output_dir="vector_store"):
        """
        Generate embeddings for the debiasing dataset and save them for fast retrieval.
        """
        print(f"Generating embeddings vector store from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} records")
        
        df = df.dropna(subset=['texto_sesgado', 'texto_corregido'])
        print(f"Records after cleaning: {len(df)}")
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        embedding_model = SentenceTransformer(self.embedding_model_name)
        
        print("Generating embeddings...")
        biased_texts = df['texto_sesgado'].tolist()
        embeddings = embedding_model.encode(biased_texts, show_progress_bar=True, batch_size=32)
        
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to {embeddings_path}")
        
        examples = []
        for _, row in df.iterrows():
            examples.append({
                'texto_sesgado': row['texto_sesgado'],
                'texto_corregido': row['texto_corregido'],
                'tipo_sesgo': row.get('tipo_sesgo', ''),
                'explicacion': row.get('explicacion', '')
            })
        
        store_path = os.path.join(output_dir, "store.pkl")
        with open(store_path, 'wb') as f:
            pickle.dump(examples, f)
        print(f"Examples saved to {store_path}")
        
        metadata = {
            'embedding_model': self.embedding_model_name,
            'num_examples': len(examples),
            'embedding_dim': embeddings.shape[1]
        }
        metadata_path = os.path.join(output_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        
        print(f"✓ Vector store created with {len(examples)} examples")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Model: {self.embedding_model_name}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Uncomment to train bias detection
    # trainer.train_bias_detection()
    
    # Uncomment to train debiasing
    # trainer.train_debiasing()
    
    # Generate embeddings vector store (run after having the dataset)
    trainer.generate_embeddings_vector_store()
