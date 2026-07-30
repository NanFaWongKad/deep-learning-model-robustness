# ============================================================
# Part 1
# Import
# Configuration
# Seed
# Device
# ============================================================


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
from torchvision.models import EfficientNet_B2_Weights


from PIL import Image


from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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


print(
    "Using device:",
    DEVICE
)



# ============================================================
# Dataset Path
# ============================================================


NORMAL_PATH = r"C:\Users\nanfa\Desktop\DoNew\TB_Chest_Radiography_Database\Normal150_1000"


TB_PATH = r"C:\Users\nanfa\Desktop\DoNew\TB_Chest_Radiography_Database\Tuberculosis150_700"



# ============================================================
# Result Path
# ============================================================


RESULT_PATH = r"C:\Users\nanfa\Desktop\DoNew\keepWork\code\result_EfficientNetB2_V2"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



# ============================================================
# Config
# ============================================================


IMG_SIZE = 150


BATCH_SIZE = 16


EPOCHS = 100


LR = 1e-3


WEIGHT_DECAY = 1e-4


NUM_FOLDS = 5


NUM_CLASSES = 2



CLASS_NAMES = [

    "Normal",

    "Tuberculosis"

]



print("Configuration loaded")

# ============================================================
# Part 2
# Dataset
# Transform
# EfficientNet-B2 Model
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
# ไม่มี Data Augmentation
# ============================================================


transform = transforms.Compose([


    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),


    transforms.ToTensor(),


    transforms.Normalize(

        IMAGENET_MEAN,

        IMAGENET_STD

    )

])



# ============================================================
# Dataset Class
# ============================================================


class TBDataset(Dataset):


    def __init__(

        self,

        transform=None

    ):


        self.images = []

        self.labels = []

        self.transform = transform



        # --------------------------
        # Normal Class = 0
        # --------------------------


        for img in os.listdir(NORMAL_PATH):


            if img.lower().endswith(

                (".png", ".jpg", ".jpeg")

            ):


                self.images.append(

                    os.path.join(

                        NORMAL_PATH,

                        img

                    )

                )


                self.labels.append(0)



        # --------------------------
        # Tuberculosis Class = 1
        # --------------------------


        for img in os.listdir(TB_PATH):


            if img.lower().endswith(

                (".png", ".jpg", ".jpeg")

            ):


                self.images.append(

                    os.path.join(

                        TB_PATH,

                        img

                    )

                )


                self.labels.append(1)





    def __len__(self):

        return len(self.images)





    def __getitem__(

        self,

        index

    ):


        img_path = self.images[index]


        label = self.labels[index]



        image = Image.open(

            img_path

        ).convert(

            "RGB"

        )



        if self.transform:


            image = self.transform(

                image

            )



        return image, label





# ============================================================
# Load Dataset
# ============================================================


dataset = TBDataset(

    transform=transform

)



labels = np.array(

    dataset.labels

)



print(
    "Total Images :",
    len(dataset)
)



print("\nClass Distribution")


for i, name in enumerate(CLASS_NAMES):


    print(

        name,

        np.sum(

            labels == i

        )

    )





# ============================================================
# EfficientNet-B2 Model
# ============================================================


def create_model():


    model = models.efficientnet_b2(

        weights = EfficientNet_B2_Weights.IMAGENET1K_V1

    )



    # --------------------------------
    # Fine-tune ทุก Layer
    # --------------------------------


    for param in model.parameters():

        param.requires_grad = True





    # --------------------------------
    # เปลี่ยน Classifier
    # --------------------------------


    in_features = model.classifier[1].in_features



    model.classifier[1] = nn.Linear(

        in_features,

        NUM_CLASSES

    )



    model = model.to(

        DEVICE

    )



    return model





print("Dataset and EfficientNet-B2 Model Ready")

# ============================================================
# Part 3
# Stratified 5-Fold Training
# EfficientNet-B2
# ============================================================


# ============================================================
# Stratified K-Fold
# ============================================================


skf = StratifiedKFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=SEED

)



# ============================================================
# Training Loop
# ============================================================


for fold, (train_idx, val_idx) in enumerate(

    skf.split(

        np.zeros(len(labels)),

        labels

    )

):


    print("\n====================================")

    print(
        f"Fold {fold+1}/{NUM_FOLDS}"
    )

    print("====================================")



    # ---------------------------------
    # Fold Result Path
    # ---------------------------------


    fold_path = os.path.join(

        RESULT_PATH,

        f"Fold_{fold+1}"

    )


    os.makedirs(

        fold_path,

        exist_ok=True

    )




    # ---------------------------------
    # Dataset Split
    # ---------------------------------


    train_subset = Subset(

        dataset,

        train_idx

    )


    val_subset = Subset(

        dataset,

        val_idx

    )





    # ---------------------------------
    # DataLoader
    # ---------------------------------


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





    # ---------------------------------
    # Create Model
    # ---------------------------------


    model = create_model()





    # ---------------------------------
    # Class Weight
    # ---------------------------------


    train_labels = labels[train_idx]



    class_count = np.bincount(

        train_labels

    )



    class_weights = (

        len(train_labels)

        /

        (

            NUM_CLASSES

            *

            class_count

        )

    )



    class_weights = torch.tensor(

        class_weights,

        dtype=torch.float

    ).to(DEVICE)




    print(

        "Class weight:",

        class_weights

    )





    # ---------------------------------
    # Loss
    # ---------------------------------


    criterion = nn.CrossEntropyLoss(

        weight=class_weights

    )





    # ---------------------------------
    # Optimizer
    # ---------------------------------


    optimizer = optim.Adam(

        model.parameters(),

        lr=LR,

        weight_decay=WEIGHT_DECAY

    )





    # ---------------------------------
    # Mixed Precision
    # ---------------------------------


    scaler = torch.cuda.amp.GradScaler()





    # ---------------------------------
    # Best Model
    # ---------------------------------


    best_auc = 0.0

    best_epoch = 0





    # ========================================================
    # Epoch Loop
    # ========================================================


    for epoch in range(EPOCHS):


        print(

            f"\nEpoch {epoch+1}/{EPOCHS}"

        )



        # ====================================================
        # Train
        # ====================================================


        model.train()



        train_loss = 0.0



        progress = tqdm(

            train_loader,

            desc="Training"

        )



        for images, targets in progress:


            images = images.to(

                DEVICE,

                non_blocking=True

            )


            targets = targets.to(

                DEVICE,

                non_blocking=True

            )



            optimizer.zero_grad()



            with torch.cuda.amp.autocast():


                outputs = model(

                    images

                )


                loss = criterion(

                    outputs,

                    targets

                )



            scaler.scale(

                loss

            ).backward()



            scaler.step(

                optimizer

            )


            scaler.update()



            train_loss += loss.item()



            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )




        train_loss /= len(train_loader)



        print(

            f"Train Loss : {train_loss:.4f}"

        )





        # ====================================================
        # Validation
        # ====================================================


        model.eval()



        y_true = []

        y_prob = []




        with torch.no_grad():


            for images, targets in val_loader:


                images = images.to(

                    DEVICE

                )



                outputs = model(

                    images

                )



                probabilities = torch.softmax(

                    outputs,

                    dim=1

                )



                y_true.extend(

                    targets.numpy()

                )



                # Probability ของ TB class

                y_prob.extend(

                    probabilities[:,1]

                    .cpu()

                    .numpy()

                )




        y_true = np.array(

            y_true

        )


        y_prob = np.array(

            y_prob

        )





        # ---------------------------------
        # Validation AUC
        # ---------------------------------


        val_auc = roc_auc_score(

            y_true,

            y_prob

        )



        print(

            f"Validation AUC : {val_auc:.4f}"

        )





        # ---------------------------------
        # Save Best Model
        # ---------------------------------


        if val_auc > best_auc:


            best_auc = val_auc


            best_epoch = epoch + 1



            torch.save(

                {


                    "epoch":

                    epoch + 1,


                    "auc":

                    val_auc,


                    "model_state_dict":

                    model.state_dict(),


                    "optimizer_state_dict":

                    optimizer.state_dict()


                },


                os.path.join(

                    fold_path,

                    "best_model.pth"

                )

            )



            print(

                "Saved Best Model"

            )




    print("\nFold Finished")

    print(

        "Best Epoch:",

        best_epoch

    )

    print(

        "Best AUC:",

        best_auc

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


        for images, labels_batch in loader:


            images = images.to(

                device

            )



            outputs = model(

                images

            )



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


            # probability class TB

            y_prob.extend(

                probabilities[:,1]

                .cpu()

                .numpy()

            )




    y_true = np.array(

        y_true

    )


    y_pred = np.array(

        y_pred

    )


    y_prob = np.array(

        y_prob

    )




    # ========================================================
    # Metrics
    # ========================================================


    accuracy = accuracy_score(

        y_true,

        y_pred

    )



    precision = precision_score(

        y_true,

        y_pred,

        zero_division=0

    )



    sensitivity = recall_score(

        y_true,

        y_pred,

        zero_division=0

    )



    f1 = f1_score(

        y_true,

        y_pred,

        zero_division=0

    )



    auc = roc_auc_score(

        y_true,

        y_prob

    )




    # ========================================================
    # Specificity
    # ========================================================


    cm = confusion_matrix(

        y_true,

        y_pred

    )


    TN = cm[0,0]

    FP = cm[0,1]

    FN = cm[1,0]

    TP = cm[1,1]



    specificity = TN / (TN + FP)



    return (

        accuracy,

        precision,

        sensitivity,

        specificity,

        f1,

        auc,

        cm

    )





# ============================================================
# Confusion Matrix Plot
# ============================================================


def save_confusion_matrix(

    cm,

    fold

):


    plt.figure(

        figsize=(5,4)

    )



    plt.imshow(

        cm,

        cmap="Blues"

    )


    plt.title(

        f"Confusion Matrix Fold {fold}"

    )


    plt.colorbar()



    classes = [

        "Normal",

        "Tuberculosis"

    ]



    plt.xticks(

        [0,1],

        classes

    )


    plt.yticks(

        [0,1],

        classes

    )



    plt.xlabel(

        "Predicted Label"

    )


    plt.ylabel(

        "True Label"

    )



    for i in range(2):


        for j in range(2):


            plt.text(

                j,

                i,

                cm[i,j],

                ha="center",

                va="center"

            )



    plt.tight_layout()



    save_path = os.path.join(

        RESULT_PATH,

        f"confusion_matrix_fold_{fold}.png"

    )


    plt.savefig(

        save_path,

        dpi=300

    )


    plt.close()






# ============================================================
# Evaluate Every Fold
# ============================================================


fold_results = []



for fold in range(NUM_FOLDS):


    print("\n==============================")

    print(

        f"Evaluate Fold {fold+1}"

    )

    print("==============================")





    # ---------------------------------
    # Get validation index
    # ---------------------------------


    _, val_idx = list(

        skf.split(

            np.zeros(len(labels)),

            labels

        )

    )[fold]





    val_subset = Subset(

        dataset,

        val_idx

    )



    val_loader = DataLoader(

        val_subset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0

    )





    # ---------------------------------
    # Load Best Model
    # ---------------------------------


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





    # ---------------------------------
    # Evaluate
    # ---------------------------------


    (

        acc,

        precision,

        sensitivity,

        specificity,

        f1,

        auc,

        cm


    ) = evaluate_model(

        model,

        val_loader,

        DEVICE

    )




    print(

        f"Accuracy    : {acc:.4f}"

    )


    print(

        f"Precision   : {precision:.4f}"

    )


    print(

        f"Sensitivity : {sensitivity:.4f}"

    )


    print(

        f"Specificity : {specificity:.4f}"

    )


    print(

        f"F1-score    : {f1:.4f}"

    )


    print(

        f"AUC         : {auc:.4f}"

    )



    print(

        "Confusion Matrix"

    )


    print(cm)




    save_confusion_matrix(

        cm,

        fold+1

    )





    fold_results.append({


        "Fold":

        fold+1,


        "Accuracy":

        acc,


        "Precision":

        precision,


        "Sensitivity":

        sensitivity,


        "Specificity":

        specificity,


        "F1-score":

        f1,


        "AUC":

        auc


    })





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


summary_df = pd.DataFrame({


    "Mean":

    results_df.mean(

        numeric_only=True

    ),



    "Std":

    results_df.std(

        numeric_only=True

    )



})





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
# Display Final Result
# ============================================================


print("\n================================")

print("Final 5-Fold Cross Validation Result")

print("================================")



print(summary_df)