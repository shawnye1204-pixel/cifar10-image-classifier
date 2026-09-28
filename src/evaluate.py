from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from . import config
from .data import test_loader
from .model import CNN


def evaluate():

    # =========================
    # Paths
    # =========================

    project_root = Path(__file__).resolve().parents[1]

    checkpoint_path = (
        project_root
        / "checkpoints"
        / config.EXPERIMENT_NAME
        / "best_accuracy_model.pth"
    )

    figures_dir = (
        project_root
        / "outputs"
        / "figures"
        / config.EXPERIMENT_NAME
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    confusion_matrix_path = (
        figures_dir / "confusion_matrix.png"
    )

    misclassified_path = (
        figures_dir / "misclassified_examples.png"
    )


    # =========================
    # Load model
    # =========================

    model = CNN(
        num_classes=config.NUM_CLASSES
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    #load the best accuracy model weights from the checkpoint
    model.load_state_dict(state_dict)

    model.eval()


    # =========================
    # Class names
    # =========================

    classes = test_loader.dataset.classes
    num_classes = len(classes)


    # =========================
    # Statistics
    # =========================

    total_correct = 0
    total_images = 0

    class_correct = torch.zeros(
        num_classes,
        dtype=torch.long
    )# Initialize a tensor to store correct predictions for each class

    class_total = torch.zeros(
        num_classes,
        dtype=torch.long
    )# Initialize a tensor to store total images for each class

    all_labels = []
    all_predictions = []

    misclassified_images = []
    misclassified_true = []
    misclassified_pred = []

    max_misclassified = 16


    # =========================
    # Test evaluation
    # =========================

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Overall accuracy
            total_correct += (
                predictions == labels
            ).sum().item()

            total_images += labels.size(0)


            # Per-class accuracy
            for class_index in range(num_classes):

                mask = labels == class_index

                class_total[class_index] += (
                    mask.sum().item()
                )

                class_correct[class_index] += (
                    predictions[mask]
                    == labels[mask]
                ).sum().item()


            # Save labels and predictions
            # for confusion matrix
            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )


            # Save some misclassified images
            wrong_indices = (
                predictions != labels
            ).nonzero(as_tuple=True)[0]

            for index in wrong_indices:

                if (
                    len(misclassified_images)
                    >= max_misclassified
                ):
                    break

                misclassified_images.append(
                    images[index].cpu()
                )

                misclassified_true.append(
                    labels[index].item()
                )

                misclassified_pred.append(
                    predictions[index].item()
                )


    # =========================
    # Overall accuracy
    # =========================

    overall_accuracy = (
        total_correct / total_images
    )

    print(
        f"Overall Test Accuracy: "
        f"{overall_accuracy:.2%}"
    )


    # =========================
    # Per-class accuracy
    # =========================

    print("\nPer-class Accuracy:")

    for class_index, class_name in enumerate(classes):

        accuracy = (
            class_correct[class_index].item()
            / class_total[class_index].item()
        )

        print(
            f"{class_name:>10}: "
            f"{accuracy:.2%}"
        )


    # =========================
    # Confusion matrix
    # =========================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes))
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(cm)

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        range(num_classes)
    )

    ax.set_yticks(
        range(num_classes)
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted Class"
    )

    ax.set_ylabel(
        "True Class"
    )

    ax.set_title(
        "CIFAR-10 Confusion Matrix"
    )

    for true_class in range(num_classes):
        for predicted_class in range(num_classes):

            ax.text(
                predicted_class,
                true_class,
                cm[
                    true_class,
                    predicted_class
                ],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # =========================
    # Common classification errors
    # =========================

    error_cm = cm.copy()

    np.fill_diagonal(
        error_cm,
        0
    )

    sorted_indices = np.argsort(
        error_cm.ravel()
    )[::-1]

    print("\nMost Common Errors:")

    shown = 0

    for flat_index in sorted_indices:

        true_class, predicted_class = (
            np.unravel_index(
                flat_index,
                error_cm.shape
            )
        )

        count = error_cm[
            true_class,
            predicted_class
        ]

        if count == 0:
            break

        print(
            f"{classes[true_class]} "
            f"→ "
            f"{classes[predicted_class]}: "
            f"{count}"
        )

        shown += 1

        if shown == 5:
            break


    # =========================
    # Misclassified examples
    # =========================

    if misclassified_images:

        rows = 4
        columns = 4

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(10, 10)
        )

        axes = axes.flatten()

        for i, ax in enumerate(axes):

            if i < len(misclassified_images):

                image = misclassified_images[i]

                # Undo Normalize(
                #     mean=(0.5, 0.5, 0.5),
                #     std=(0.5, 0.5, 0.5)
                # )
                image = image * 0.5 + 0.5

                image = image.permute(
                    1,
                    2,
                    0
                )

                image = image.clamp(
                    0,
                    1
                )

                ax.imshow(
                    image.numpy()
                )

                true_name = classes[
                    misclassified_true[i]
                ]

                predicted_name = classes[
                    misclassified_pred[i]
                ]

                ax.set_title(
                    f"True: {true_name}\n"
                    f"Pred: {predicted_name}"
                )

            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            misclassified_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()


    print(
        f"\nSaved: {confusion_matrix_path}"
    )

    print(
        f"Saved: {misclassified_path}"
    )


if __name__ == "__main__":
    evaluate()