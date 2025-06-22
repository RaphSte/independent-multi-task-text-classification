import json
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, callbacks

class AnimalFactModelTrainer:
    def __init__(self, df, embedding_column="embedding"):
        """
        Initializes the trainer with a DataFrame containing animal facts.
        This DataFrame is expected to have precomputed text embeddings.

        Parameters:
            df (pd.DataFrame): DataFrame with animal facts.
            embedding_column (str): Name of the column containing the embeddings.
        """
        self.df = df
        self.embedding_column = embedding_column

    def train_model(self, target_column,
                    model_name_prefix="animal_model",
                    epochs=50,
                    batch_size=64,
                    patience=5):
        """
        Trains a neural network on the entire dataset (without splitting into train/test)
        to predict the target category from the text embeddings.

        The function also remaps the target labels to contiguous integers and saves the
        trained model and the mapping to disk.

        Parameters:
            target_column (str): The column in df to predict.
            model_name_prefix (str): Prefix for the saved model and mapping files.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            patience (int): Patience for the early stopping callback.

        Returns:
            model: The trained Keras model.
            history: The training history.
            index_to_label (dict): Mapping of predicted indices to original labels.
        """
        print(f"Training model to predict: {target_column}")

        # Load text embeddings and target labels from the DataFrame.
        # Each embedding should be a list of length 1536.
        X = np.array(self.df[self.embedding_column].tolist(), dtype=np.float32)
        y = np.array(self.df[target_column])

        # Remap label values to contiguous indices.
        unique_labels = np.sort(np.unique(y))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}

        # Map the target labels.
        y_mapped = np.array([label_to_index[label] for label in y])
        num_classes = len(unique_labels)
        print("Number of classes:", num_classes)
        print("Using all data for training.\n")

        input_dim = X.shape[1]

        # Define the neural network.
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(2048, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Early stopping callback using a small validation_split (e.g., 10%)
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        history = model.fit(X, y_mapped,
                            validation_split=0.1,  # a small fraction for early stopping monitoring
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stop])

        model.summary()

        # Save the trained model.
        model_filename = f"{model_name_prefix}_{target_column}.keras"
        model.save(model_filename)
        print(f"Model saved to {model_filename}")

        # Save the mapping of indices to original labels.
        mapping_filename = f"{model_name_prefix}_{target_column}_mapping.json"
        with open(mapping_filename, "w") as f:
            json.dump(index_to_label, f)
        print(f"Mapping saved to {mapping_filename}\n")

        return model, history, index_to_label
