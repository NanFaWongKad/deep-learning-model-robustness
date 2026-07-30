# ============================================================
# Part 1
# Import
# Configuration
# Seed
# Device
# ============================================================
from torch.amp import autocast, GradScaler

import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from PIL import Image

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc as sklearn_auc
)

from tqdm import tqdm

import matplotlib.pyplot as plt

# ============================================================
# Seed
# ============================================================

SEED = 42

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

# ============================================================
# Dataset Path
# ============================================================

IMAGE_PATH = r"C:\Users\nanfa\Desktop\DoNew\rsna-pneumonia-detection-challenge\trainset224"

CSV_PATH = r"C:\Users\nanfa\Desktop\DoNew\rsna-pneumonia-detection-challenge\stage_2_detailed_class_info.csv"

# ============================================================
# Result Path
# ============================================================

RESULT_PATH = r"C:\Users\nanfa\Desktop\DoNew\keepWork\code\result"

os.makedirs(
    RESULT_PATH,
    exist_ok=True
)

# ============================================================
# Config
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 64

ACCUMULATION_STEPS = 1

EPOCHS = 30

LR = 5e-4

WEIGHT_DECAY = 1e-4

NUM_FOLDS = 5

NUM_CLASSES = 3

CLASS_NAMES = [

    "Normal",

    "Lung Opacity",

    "No Lung Opacity / Not Normal"

]

print("Configuration Loaded")

print(f"Image Size           : {IMG_SIZE}")
print(f"Batch Size           : {BATCH_SIZE}")
print(f"Gradient Accumulation: {ACCUMULATION_STEPS}")
print(f"Effective Batch Size : {BATCH_SIZE * ACCUMULATION_STEPS}")
print(f"Epochs               : {EPOCHS}")
print(f"Learning Rate        : {LR}")
print(f"Optimizer            : AdamW")
print(f"Loss Function        : CrossEntropyLoss")
print(f"5-Fold CV            : Yes")
print(f"Data Leakage         : Prevent by PatientID")

# ============================================================
# Part 2
# Dataset
# Transform
# ResNet50 Model
# No Data Augmentation
# ============================================================


# ============================================================
# ImageNet Normalization
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# Transform
# No Data Augmentation
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        IMAGENET_MEAN,

        IMAGENET_STD

    )

])


# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# Remove duplicate patientId
df = df.drop_duplicates(
    subset=["patientId"]
).reset_index(drop=True)

# ============================================================
# Label Mapping
# ============================================================

label_mapping = {

    "Normal": 0,

    "Lung Opacity": 1,

    "No Lung Opacity / Not Normal": 2

}


df["label"] = df["class"].map(label_mapping)


# ============================================================
# Create Image Path
# patientId.png
# ============================================================

df["image_path"] = df["patientId"].apply(

    lambda x: os.path.join(

        IMAGE_PATH,

        f"{x}.png"

    )

)


# ============================================================
# Check Image Matching
# ============================================================

missing_images = df[

    ~df["image_path"].apply(os.path.exists)

]


print("\nMissing Images :", len(missing_images))


if len(missing_images) > 0:

    print(

        missing_images["patientId"].head()

    )


# ============================================================
# Remove Missing Images
# ============================================================

df = df[

    df["image_path"].apply(os.path.exists)

].reset_index(drop=True)

# ============================================================
# Duplicate Patient Statistics
# ============================================================

counts = df["patientId"].value_counts()

print("\nPatient Statistics")

print("Total Patients :", len(counts))

print("Patients with >1 image :", (counts > 1).sum())

print("Extra duplicate images :", (counts - 1).clip(lower=0).sum())

print("Maximum images per patient :", counts.max())

print("\nImages per Patient Distribution")
print(counts.value_counts().sort_index())

# ============================================================
# Dataset Class
# ============================================================

class RSNADataset(Dataset):

    def __init__(

        self,

        dataframe,

        transform=None

    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.transform = transform


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(

        self,

        index

    ):

        row = self.dataframe.iloc[index]


        image = Image.open(

            row["image_path"]

        ).convert(

            "RGB"

        )


        if self.transform:

            image = self.transform(

                image

            )


        label = int(

            row["label"]

        )


        patient = row["patientId"]


        return (

            image,

            label,

            patient

        )



# ============================================================
# Create Dataset
# ============================================================

dataset = RSNADataset(

    df,

    transform=transform

)


# ============================================================
# Labels and Groups
# For Stratified Group K-Fold
# ============================================================

labels = df["label"].values

groups = df["patientId"].values



# ============================================================
# Dataset Information
# ============================================================

print("\nTotal Images :", len(dataset))

print(

    "Total Patients :",

    df["patientId"].nunique()

)



print("\nClass Distribution")


for i, name in enumerate(CLASS_NAMES):

    print(

        name,

        np.sum(labels == i)

    )



# ============================================================
# Duplicate Patient Check
# ============================================================

duplicate_patient = (

    df["patientId"].duplicated().sum()

)


print(

    "\nDuplicate Patient Images :", 

    duplicate_patient

)



# ============================================================
# ResNet50 Model
# ============================================================

def create_model():


    model = models.resnet50(

        weights=ResNet50_Weights.IMAGENET1K_V2

    )


    # --------------------------------
    # Fine Tune All Layers
    # --------------------------------

    for param in model.parameters():

        param.requires_grad = True



    # --------------------------------
    # Replace FC Layer
    # --------------------------------

    in_features = model.fc.in_features


    model.fc = nn.Linear(

        in_features,

        NUM_CLASSES

    )


    model = model.to(

        DEVICE

    )


    return model



print(

    "Dataset and ResNet50 Model Ready"

)

# ============================================================
# Part 3
# 5-Fold Cross Validation
# Training
# Class Weight Loss
# ============================================================


from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# Cross Validation Setting
# ============================================================

NUM_FOLDS = 5


skf = StratifiedGroupKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=SEED

)



# ============================================================
# Training Configuration
# ============================================================

# ============================================================
# Store Results
# ============================================================

fold_results = []



# ============================================================
# Start 5 Fold Training
# ============================================================

for fold, (train_idx, val_idx) in enumerate(

    skf.split(

        df,

        labels,

        groups

    )

):


    print("\n")

    print("="*60)

    print(

        f"Fold {fold+1}/{NUM_FOLDS}"

    )

    print("="*60)

    train_percent = len(train_idx) / len(df) * 100
    val_percent = len(val_idx) / len(df) * 100

    train_patients = df.iloc[train_idx]["patientId"].nunique()
    val_patients = df.iloc[val_idx]["patientId"].nunique()

    print(f"Train Images      : {len(train_idx)} ({train_percent:.2f}%)")
    print(f"Validation Images : {len(val_idx)} ({val_percent:.2f}%)")

    print(f"Train Patients    : {train_patients}")
    print(f"Validation Patients : {val_patients}")

    # --------------------------------------------------------
    # Dataset Split
    # --------------------------------------------------------

    train_subset = Subset(

        dataset,

        train_idx

    )


    val_subset = Subset(

        dataset,

        val_idx

    )



    train_loader = DataLoader(

        train_subset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True

    )


    val_loader = DataLoader(

        val_subset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )



    # ========================================================
    # Class Weight
    # ========================================================

    train_labels = labels[train_idx]


    class_counts = np.bincount(

        train_labels,

        minlength=NUM_CLASSES

    )


    class_weights = (

        len(train_labels)

        /

        (

            NUM_CLASSES

            *

            class_counts

        )

    )


    class_weights = torch.tensor(

        class_weights,

        dtype=torch.float32

    ).to(DEVICE)



    print(

        "Class Weight:",

        class_weights.cpu().numpy()

    )



    # ========================================================
    # Model
    # ========================================================

    model = create_model()



    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights

    )



    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.Adam(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY

    )
    scaler = GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )


    # ========================================================
    # Best AUC
    # ========================================================

    best_auc = 0.0


    best_model_path = os.path.join(

    RESULT_PATH,

    f"Fold_{fold+1}",

    "best_model.pth"

    )
    os.makedirs(

    os.path.dirname(best_model_path),

    exist_ok=True

    )


    # ========================================================
    # Epoch Training
    # ========================================================

    for epoch in range(EPOCHS):


        model.train()


        train_loss = 0.0


        train_correct = 0

        train_total = 0



        for images, targets, _ in train_loader:


            images = images.to(DEVICE)

            targets = targets.to(DEVICE)



            optimizer.zero_grad()


            with autocast("cuda"):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )


            scaler.scale(loss).backward()

            scaler.step(
                optimizer
            )

            scaler.update()



            train_loss += loss.item()



            _, predicted = torch.max(

                outputs,

                1

            )


            train_total += targets.size(0)


            train_correct += (

                predicted == targets

            ).sum().item()



        train_acc = (

            train_correct /

            train_total

        )



        # ====================================================
        # Validation
        # ====================================================

        model.eval()


        val_targets = []

        val_predictions = []

        val_probabilities = []



        with torch.no_grad():


            for images, targets, _ in val_loader:


                images = images.to(DEVICE)


                outputs = model(images)



                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )



                _, predicted = torch.max(

                    outputs,

                    1

                )



                val_targets.extend(

                    targets.numpy()

                )


                val_predictions.extend(

                    predicted.cpu().numpy()

                )


                val_probabilities.extend(

                    probabilities.cpu().numpy()

                )



        val_auc = roc_auc_score(

            val_targets,

            np.array(val_probabilities),

            multi_class="ovr",

            average="macro"

        )



        val_acc = accuracy_score(

            val_targets,

            val_predictions

        )



        print(

            f"Epoch [{epoch+1}/{EPOCHS}] "

            f"Train Acc: {train_acc:.4f} "

            f"Val Acc: {val_acc:.4f} "

            f"Val AUC: {val_auc:.4f}"

        )



        # ====================================================
        # Save Best Model by Validation AUC
        # ====================================================

        if val_auc > best_auc:


            best_auc = val_auc

            torch.save(

                {

                    "model_state_dict": model.state_dict(),

                    "best_auc": best_auc

                },

                best_model_path

            )


            print(

                "Best Model Saved"

            )



    # ========================================================
    # Fold Result
    # ========================================================

    fold_results.append({

        "Fold": fold + 1,

        "Best AUC": best_auc

    })



# ============================================================
# Summary
# ============================================================

results_df = pd.DataFrame(

    fold_results

)


print("\n")

print(results_df)


print(

    "\nMean AUC:",

    results_df["Best AUC"].mean()

)


print(

    "Std AUC:",

    results_df["Best AUC"].std()

)

# ============================================================
# Part 4
# Evaluation
# Best Model Evaluation
# Confusion Matrix
# ============================================================

# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(

    model,

    loader,

    device

):

    model.eval()

    y_true = []

    y_pred = []

    y_prob = []

    with torch.no_grad():

        for images, labels_batch, _ in loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(

                outputs,

                dim=1

            )

            predictions = torch.argmax(

                probabilities,

                dim=1

            )

            y_true.extend(

                labels_batch.numpy()

            )

            y_pred.extend(

                predictions.cpu().numpy()

            )

            y_prob.extend(

                probabilities.cpu().numpy()

            )

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    y_prob = np.array(y_prob)

    y_true_bin = label_binarize(

        y_true,

        classes=[0,1,2]

    )

    accuracy = accuracy_score(

        y_true,

        y_pred

    )

    precision = precision_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    recall = recall_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    f1 = f1_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0

    )

    macro_auc = roc_auc_score(

        y_true_bin,

        y_prob,

        multi_class="ovr",

        average="macro"

    )

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    return (

        accuracy,

        precision,

        recall,

        f1,

        macro_auc,

        cm,

        y_true,

        y_pred,

        y_prob

    )


# ============================================================
# Save Confusion Matrix
# ============================================================

def save_confusion_matrix(

    cm,

    fold

):

    plt.figure(

        figsize=(6,6)

    )

    plt.imshow(

        cm,

        cmap="Blues"

    )

    plt.title(

        f"Confusion Matrix Fold {fold}"

    )

    plt.colorbar()

    plt.xticks(

        np.arange(NUM_CLASSES),

        CLASS_NAMES,

        rotation=20

    )

    plt.yticks(

        np.arange(NUM_CLASSES),

        CLASS_NAMES

    )

    plt.xlabel(

        "Predicted Label"

    )

    plt.ylabel(

        "True Label"

    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(

                j,

                i,

                cm[i,j],

                ha="center",

                va="center",

                color="black"

            )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            RESULT_PATH,

            f"confusion_matrix_fold_{fold}.png"

        ),

        dpi=300

    )

    plt.close()

# ============================================================
# ROC Curve Function
# ============================================================

def save_roc_curve(
    y_true,
    y_prob,
    fold
):


    y_true_bin = label_binarize(
        y_true,
        classes=[0,1,2]
    )


    plt.figure(
        figsize=(8,6)
    )


    # ----------------------------
    # Per Class ROC
    # ----------------------------

    class_auc_results = []


    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:,i],
            y_prob[:,i]
        )


        roc_auc = sklearn_auc(
            fpr,
            tpr
        )


        class_auc_results.append({

            "Class": CLASS_NAMES[i],

            "AUC": roc_auc

        })


        plt.plot(
            fpr,
            tpr,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.4f})"
        )

    # ----------------------------
    # Random Guess Line
    # ----------------------------

    plt.plot(

        [0,1],

        [0,1],

        linestyle=":"

    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        f"ROC Curve Fold {fold}"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid()


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            RESULT_PATH,

            f"roc_curve_fold_{fold}.png"

        ),

        dpi=300

    )


    plt.close()



    # ----------------------------
    # Save AUC Per Class
    # ----------------------------

    auc_df = pd.DataFrame(

        class_auc_results

    )


    auc_df.to_csv(

        os.path.join(

            RESULT_PATH,

            f"auc_per_class_fold_{fold}.csv"

        ),

        index=False

    )



    return class_auc_results

# ============================================================
# Overall ROC Storage (OOF Prediction)
# ============================================================

overall_y_true = []

overall_y_prob = []

# ============================================================
# Overall ROC Curve
# ============================================================

def save_overall_roc_curve(
    y_true,
    y_prob
):

    y_true = np.array(y_true)

    y_prob = np.array(y_prob)


    y_true_bin = label_binarize(
        y_true,
        classes=[0,1,2]
    )


    plt.figure(
        figsize=(8,6)
    )


    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:,i],
            y_prob[:,i]
        )


        roc_auc = sklearn_auc(
            fpr,
            tpr
        )


        plt.plot(
            fpr,
            tpr,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.4f})"
        )


    macro_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        multi_class="ovr",
        average="macro"
    )


    plt.plot(
        [0,1],
        [0,1],
        linestyle=":"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        f"Overall ROC Curve (5-Fold CV) Macro AUC={macro_auc:.4f}"
    )


    plt.legend(
        loc="lower right"
    )


    plt.grid()


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            RESULT_PATH,
            "overall_roc_curve.png"
        ),
        dpi=300
    )


    plt.close()


    return macro_auc

# ============================================================
# Evaluate Every Fold
# ============================================================

fold_results = []

for fold, (

    train_idx,

    val_idx

) in enumerate(

    skf.split(

        df,

        labels,

        groups

    )

):

    print("\n========================================")

    print(f"Evaluate Fold {fold+1}")

    print("========================================")

    val_subset = Subset(

        dataset,

        val_idx

    )

    val_loader = DataLoader(

        val_subset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )

    model = create_model()

    checkpoint = torch.load(
        os.path.join(
            RESULT_PATH,
            f"Fold_{fold+1}",
            "best_model.pth"
        ),
        map_location=DEVICE
    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.to(

        DEVICE

    )

    (

        acc,

        precision,

        recall,

        f1,

        macro_auc,

        cm,

        y_true,

        y_pred,

        y_prob

    ) = evaluate_model(

        model,

        val_loader,

        DEVICE

    )

    # ====================================================
    # Save Out-of-Fold Prediction for Overall ROC
    # ====================================================

    overall_y_true.extend(
        y_true
    )

    overall_y_prob.extend(
        y_prob
    )

    print(f"Accuracy  : {acc:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1-score  : {f1:.4f}")

    print(f"AUC       : {macro_auc:.4f}")

    print("\nConfusion Matrix")

    print(cm)

    save_confusion_matrix(

        cm,

        fold+1

    )

    class_auc = save_roc_curve(

        y_true,

        y_prob,

        fold+1

    )


    print("\nClass AUC")

    for item in class_auc:

        print(
            item["Class"],
            ":",
            round(item["AUC"],4)
        )

    prediction_df = pd.DataFrame({

        "True Label": y_true,

        "Predicted Label": y_pred

    })

    prediction_df.to_csv(

        os.path.join(

            RESULT_PATH,

            f"Fold_{fold+1}",

            "prediction.csv"

        ),

        index=False

    )

    fold_results.append({

        "Fold": fold+1,

        "Accuracy": acc,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1,

        "Macro AUC": macro_auc,

        "Normal AUC": class_auc[0]["AUC"],

        "Lung Opacity AUC": class_auc[1]["AUC"],

        "No Lung Opacity AUC": class_auc[2]["AUC"]   

    })

# ============================================================
# Generate Overall ROC
# ============================================================

overall_macro_auc = save_overall_roc_curve(

    overall_y_true,

    overall_y_prob

)


print(
    "\nOverall Macro AUC:",
    round(overall_macro_auc,4)
)


print("\nEvaluation Complete")

# ============================================================
# Part 5
# Save Fold Results
# Mean ± Std Summary
# ============================================================

# ============================================================
# Save Fold Results CSV
# ============================================================

results_df = pd.DataFrame(

    fold_results

)

results_csv = os.path.join(

    RESULT_PATH,

    "fold_results.csv"

)

results_df.to_csv(

    results_csv,

    index=False

)

print(

    "Saved:",

    results_csv

)

# ============================================================
# Mean ± Std
# ============================================================

metric_columns = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1-score",

    "Macro AUC",

    "Normal AUC",

    "Lung Opacity AUC",

    "No Lung Opacity AUC"
   

]

summary_df = pd.DataFrame(

    index=metric_columns,

    columns=[

        "Mean",

        "Std"

    ]

)

for metric in metric_columns:

    summary_df.loc[metric, "Mean"] = results_df[metric].mean()

    summary_df.loc[metric, "Std"] = results_df[metric].std()

summary_csv = os.path.join(

    RESULT_PATH,

    "summary_results.csv"

)

summary_df.to_csv(

    summary_csv

)

print(

    "Saved:",

    summary_csv

)

# ============================================================
# Best Fold
# ============================================================

best_fold = results_df.loc[

    results_df["Macro AUC"].idxmax()

]

best_fold.to_frame().to_csv(

    os.path.join(

        RESULT_PATH,

        "best_fold.csv"

    ),

    header=False

)

print(

    "\nBest Fold"

)

print(best_fold)

# ============================================================
# Save All Results (JSON)
# ============================================================

with open(

    os.path.join(

        RESULT_PATH,

        "all_results.json"

    ),

    "w"

) as f:

    json.dump(

        fold_results,

        f,

        indent=4

    )

# ============================================================
# Display Final Result
# ============================================================

print("\n===================================================")
print("Final 5-Fold Cross Validation Result")
print("===================================================")

print(results_df)

print("\n===================================================")
print("Mean ± Std")
print("===================================================")

print(summary_df)

# ============================================================
# Finish
# ============================================================

print("\nAll Results Saved Successfully")

print("\nResult Folder")

print(RESULT_PATH)
