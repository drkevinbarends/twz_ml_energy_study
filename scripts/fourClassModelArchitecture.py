import logging
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, auc
from sklearn.preprocessing import label_binarize


# -------------------------
# Reproducibility
# -------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Model:

    def __init__(
        self,
        in_shape,
        x_train,
        learning_rate=5e-5,
        layer_sizes=None,
        dropout_rate=0.0,
        use_batchnorm=False
    ):

        # -------------------------
        # Normalization layer
        # -------------------------
        norm = Normalization()
        norm.adapt(x_train)

        layers = [
            Input(shape=(in_shape,)),
            norm
        ]

        # -------------------------
        # Hidden layers
        # -------------------------
        for width in layer_sizes:

            layers.append(Dense(width, activation="relu"))

            if use_batchnorm:
                layers.append(BatchNormalization())

            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))

        # -------------------------
        # Output layer
        # -------------------------
        layers.append(Dense(4, activation="softmax"))

        self.model = Sequential(layers)

        # -------------------------
        # Compile model
        # -------------------------
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            weighted_metrics=[]
        )


    def train(
        self,
        x_train,
        y_train,
        weights_train,
        x_test,
        y_test,
        weights_test,
        class_weights,
        batch_size,
        epochs,
        output_dir
    ):

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            x_train,
            y_train,
            sample_weight=weights_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test, weights_test),
            class_weight=None,
            callbacks=[early_stopping],
            verbose=1
        )

        self.evaluate(
            x_train,
            y_train,
            x_test,
            y_test,
            weights_train,
            weights_test,
            history,
            output_dir
        )


    def evaluate(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        weights_train,
        weights_test,
        history,
        output_dir
    ):

        y_pred_prob = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
        logging.info(f"Accuracy: {accuracy:.4f}")

        y_test_oh = label_binarize(y_test, classes=[0, 1, 2, 3])

        aucs = []

        for i in range(4):

            fpr, tpr, _ = roc_curve(
                y_test_oh[:, i],
                y_pred_prob[:, i],
                sample_weight=weights_test
            )

            # enforce monotonicity
            fpr = np.maximum.accumulate(fpr)

            auc = np.trapz(tpr, fpr)

            aucs.append(auc)

        roc_auc = np.mean(aucs)

        logging.info(f"Macro ROC-AUC (OvR): {roc_auc:.4f}")

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["other", "ZZ", "ttZ", "tWZ"],
                sample_weight=weights_test
            )
        )

        self.plot_loss(
            history, 
            output_dir
        )
        self.plot_roc_curve(
            y_test, 
            y_pred_prob, 
            weights_test, 
            output_dir
        )
        self.plot_train_vs_val_roc(
            x_train,
            y_train,
            x_test,
            y_test,
            weights_train,
            weights_test,
            output_dir
        )


    def plot_loss(self, history, output_dir):

        plt.figure(figsize=(10, 7))

        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")

        plt.legend()
        plt.savefig(f"{output_dir}/Training_LossPlot.png")
        plt.close()


    def plot_roc_curve(self, y_test, y_pred_prob, weights_test, output_dir):

        y_test_oh = label_binarize(y_test, classes=[0,1,2,3])
        class_names = ["other", "ZZ", "ttZ", "tWZ"]

        plt.figure(figsize=(10,7))

        for i, name in enumerate(class_names):

            fpr, tpr, _ = roc_curve(
                y_test_oh[:, i],
                y_pred_prob[:, i],
                sample_weight=weights_test
            )

            fpr = np.maximum.accumulate(fpr)

            auc = np.trapz(tpr, fpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        plt.plot([0,1],[0,1],"k--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC (OvR)")
        plt.legend()

        plt.savefig(f"{output_dir}/Training_RocCurve.png")
        plt.close()

    def plot_train_vs_val_roc(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        weights_train,
        weights_test,
        output_dir
    ):

        class_names = ["other", "ZZ", "ttZ", "tWZ"]

        y_train_prob = self.model.predict(x_train)
        y_test_prob = self.model.predict(x_test)

        y_train_oh = label_binarize(y_train, classes=[0,1,2,3])
        y_test_oh = label_binarize(y_test, classes=[0,1,2,3])

        fig, axes = plt.subplots(2,2, figsize=(12,10))
        axes = axes.flatten()

        for i, name in enumerate(class_names):

            ax = axes[i]

            # Training ROC (unweighted)
            fpr_train, tpr_train, _ = roc_curve(
                y_train_oh[:, i],
                y_train_prob[:, i],
                sample_weight=weights_train
            )

            # fpr_train = np.maximum.accumulate(fpr_train)
            # auc_train = np.trapz(tpr_train, fpr_train)
            auc_train = auc(fpr_train, tpr_train)

            # Validation ROC (weighted)
            fpr_val, tpr_val, _ = roc_curve(
                y_test_oh[:, i],
                y_test_prob[:, i],
                sample_weight=weights_test
            )

            # fpr_val = np.maximum.accumulate(fpr_val)
            # auc_val = np.trapz(tpr_val, fpr_val)
            auc_val = auc(fpr_val, tpr_val)

            ax.plot(
                fpr_train,
                tpr_train,
                label=f"Train (AUC={auc_train:.3f})",
                color="blue"
            )

            ax.plot(
                fpr_val,
                tpr_val,
                label=f"Validation (AUC={auc_val:.3f})",
                color="red"
            )

            ax.plot([0,1],[0,1],'k--')

            ax.set_title(name)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")

            ax.legend()

        plt.tight_layout()

        plt.savefig(f"{output_dir}/Train_vs_Validation_ROC.png")
        plt.close()

    def save_model(self, output_dir):

        self.model.save(f"{output_dir}/model.keras")