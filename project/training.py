import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class ModelTrainer:
    """
    Train a text classification model to detect biases in job postings.
    """

    def __init__(self, base_model_name="dccuchile/bert-base-spanish-wwm-cased", generated_model_name="my_model", dataset_path="dataset.csv"):
        self.base_model_name = base_model_name
        self.generated_model_name = generated_model_name
        self.dataset_path = dataset_path

    def load_data(self):
        """
        Load and preprocess the dataset.
        """
        df = pd.read_csv(self.dataset_path)

        # Remove rows with missing label_id
        print(f"Total records before cleaning: {len(df)}")
        df = df.dropna(subset=['label_id'])
        print(f"Total records after removing NaN labels: {len(df)}")
        
        # Use the label_id column directly as labels
        df['label'] = df['label_id'].astype(int)
        id2label = {0: 'sesgo_genero_masculino', 1: 'sesgo_genero_femenino', 2: 'sesgo_edad_joven', 3: 'sesgo_edad_mayor', 4: 'sesgo_nacionalidad', 5: 'neutral_inclusivo'}
        
        print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

        # Split the dataset
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        # Convert pandas dataframes to Hugging Face Datasets
        dataset_train = Dataset.from_pandas(train_df)
        dataset_test = Dataset.from_pandas(test_df)

        return dataset_train, dataset_test, id2label
    
    def train(self):
        """
        Train the model.
        """
        dataset_train, dataset_test, id2label = self.load_data()

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        def tokenize_function(examples):
            return tokenizer(examples["texto"], padding="max_length", truncation=True, max_length=128)

        encoded_dataset_train = dataset_train.map(tokenize_function, batched=True)
        encoded_dataset_test = dataset_test.map(tokenize_function, batched=True)

        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(id2label)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name, 
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset_train,
            eval_dataset=encoded_dataset_test,
        )

        print("Starting training...")
        trainer.train()

        print("Evaluating...")
        metrics = trainer.evaluate()
        print(metrics)

        # 8. Save the model for later use
        model.save_pretrained(f"./{self.generated_model_name}")
        tokenizer.save_pretrained(f"./{self.generated_model_name}")
        print("Model saved!")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
