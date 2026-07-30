# ============================================================
# VGG16 Transfer Learning
# COVID-19 Radiography Dataset
# Stratified 5-Fold Cross Validation
# PyTorch
# ============================================================


import os
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import models, transforms
from torchvision.models import VGG16_Weights

from PIL import Image

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

from tqdm import tqdm



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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("Using device:", DEVICE)



# ============================================================
# Paths
# ============================================================


COVID_PATH = r"C:\Users\nanfa\Desktop\DoNew\COVID-19_Radiography_Dataset\COVID224"

NORMAL_PATH = r"C:\Users\nanfa\Desktop\DoNew\COVID-19_Radiography_Dataset\Normal224"

VIRAL_PATH = r"C:\Users\nanfa\Desktop\DoNew\COVID-19_Radiography_Dataset\Viral Pneumonia224"



RESULT_PATH = r"C:\Users\nanfa\Desktop\DoNew\keepNASNet\code\result"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)



# ============================================================
# Config
# ============================================================


IMG_SIZE = 224

BATCH_SIZE = 32

ACCUMULATION_STEP = 2
# effective batch = 64


EPOCHS = 50


LR = 1e-5


NUM_FOLDS = 5



CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia"
]

NUM_CLASSES = 3



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
# ============================================================


# Training augmentation
# COVID + Viral เท่านั้น


train_transform_aug = transforms.Compose([

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])



# Normal ไม่มี rotation

train_transform_normal = transforms.Compose([


    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])



# Validation/Test

test_transform = transforms.Compose([


    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )

])





# ============================================================
# Custom Dataset
# ============================================================


class COVIDDataset(Dataset):


    def __init__(self):


        self.images = []

        self.labels = []

        self.transforms = []



        # ----------------------
        # COVID label 0
        # ----------------------

        for img in os.listdir(COVID_PATH):

            if img.lower().endswith(
                (".png",".jpg",".jpeg")
            ):

                self.images.append(
                    os.path.join(
                        COVID_PATH,
                        img
                    )
                )

                self.labels.append(0)

                self.transforms.append(
                    train_transform_aug
                )



        # ----------------------
        # Normal label 1
        # ----------------------

        for img in os.listdir(NORMAL_PATH):

            if img.lower().endswith(
                (".png",".jpg",".jpeg")
            ):

                self.images.append(
                    os.path.join(
                        NORMAL_PATH,
                        img
                    )
                )

                self.labels.append(1)

                self.transforms.append(
                    train_transform_normal
                )



        # ----------------------
        # Viral label 2
        # ----------------------

        for img in os.listdir(VIRAL_PATH):

            if img.lower().endswith(
                (".png",".jpg",".jpeg")
            ):

                self.images.append(
                    os.path.join(
                        VIRAL_PATH,
                        img
                    )
                )

                self.labels.append(2)

                self.transforms.append(
                    train_transform_aug
                )



    def __len__(self):

        return len(self.images)




    def __getitem__(self,index):


        img_path = self.images[index]

        label = self.labels[index]


        image = Image.open(
            img_path
        ).convert(
            "RGB"
        )


        image = self.transforms[index](
            image
        )


        return image, label





# ============================================================
# Load Dataset
# ============================================================


dataset = COVIDDataset()



print(
    "Total images:",
    len(dataset)
)



labels = np.array(
    dataset.labels
)



print(
    "Class distribution:"
)


for i,c in enumerate(CLASS_NAMES):

    print(
        c,
        np.sum(labels==i)
    )





# ============================================================
# VGG16 Model
# ============================================================


def create_model():


    model = models.vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1
    )


    # Freeze block1-block3

    for name,param in model.features.named_parameters():


        block = int(
            name.split(".")[0]
        )


        if block <= 16:

            param.requires_grad=False



    # classifier final layer

    model.classifier[6] = nn.Linear(
        4096,
        NUM_CLASSES
    )



    model = model.to(
        DEVICE
    )


    return model





# ============================================================
# End Part 1
# ============================================================
# ============================================================
# Stratified 5 Fold Cross Validation
# ============================================================


skf = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=SEED
)



all_fold_results = []



# ============================================================
# Fold Training
# ============================================================


for fold,(train_idx,val_idx) in enumerate(
    skf.split(
        np.zeros(len(labels)),
        labels
    )
):


    print("\n==============================")
    print(
        f"Fold {fold+1}/{NUM_FOLDS}"
    )
    print("==============================")



    fold_path = os.path.join(
        RESULT_PATH,
        f"Fold_{fold+1}"
    )


    os.makedirs(
        fold_path,
        exist_ok=True
    )



    # ----------------------------
    # Dataset split
    # ----------------------------


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



    # ----------------------------
    # Model
    # ----------------------------


    model = create_model()



    # ----------------------------
    # Loss
    # ----------------------------


    # class weight
    # ลดปัญหา class imbalance


    train_labels = labels[train_idx]


    class_count = np.bincount(
        train_labels
    )


    class_weights = (
        len(train_labels) /
        (
            NUM_CLASSES *
            class_count
        )
    )


    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(DEVICE)



    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )



    # ----------------------------
    # Optimizer
    # ----------------------------


    optimizer = optim.Adam(
        filter(
            lambda p:p.requires_grad,
            model.parameters()
        ),
        lr=LR,
        weight_decay=1e-4
    )



    # AMP
    scaler = torch.cuda.amp.GradScaler()



    best_auc = 0

    best_epoch = 0



    # ====================================================
    # Training
    # ====================================================


    for epoch in range(EPOCHS):


        model.train()


        running_loss = 0

        optimizer.zero_grad()



        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )



        for step,(images,targets) in enumerate(loop):


            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )



            with torch.cuda.amp.autocast():


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    targets
                )


                # gradient accumulation

                loss = (
                    loss /
                    ACCUMULATION_STEP
                )



            scaler.scale(
                loss
            ).backward()



            if (
                (step+1)
                %
                ACCUMULATION_STEP
                ==
                0
            ):


                scaler.step(
                    optimizer
                )


                scaler.update()


                optimizer.zero_grad()



            running_loss += loss.item()



        print(
            "Loss:",
            running_loss / len(train_loader)
        )



        # =================================================
        # Validation
        # =================================================


        model.eval()


        val_pred=[]

        val_true=[]

        val_prob=[]



        with torch.no_grad():


            for images,targets in val_loader:


                images = images.to(
                    DEVICE
                )


                outputs = model(
                    images
                )


                probs = torch.softmax(
                    outputs,
                    dim=1
                )


                _,pred = torch.max(
                    outputs,
                    1
                )



                val_pred.extend(
                    pred.cpu().numpy()
                )


                val_true.extend(
                    targets.numpy()
                )


                val_prob.extend(
                    probs.cpu().numpy()
                )



        val_true = np.array(
            val_true
        )

        val_pred = np.array(
            val_pred
        )

        val_prob = np.array(
            val_prob
        )



        auc = roc_auc_score(
            label_binarize(
                val_true,
                classes=[
                    0,1,2
                ]
            ),
            val_prob,
            multi_class="ovr",
            average="macro"
        )



        print(
            "Validation AUC:",
            auc
        )



        # save best model


        if auc > best_auc:


            best_auc = auc

            best_epoch = epoch+1



            torch.save(
                {
                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "epoch":
                    epoch+1,

                    "auc":
                    auc

                },
                os.path.join(
                    fold_path,
                    "best_model.pth"
                )
            )



    # ====================================================
    # Final Evaluation
    # ====================================================


    print(
        "Evaluate Fold",
        fold+1
    )



    model.load_state_dict(

        torch.load(
            os.path.join(
                fold_path,
                "best_model.pth"
            ),
            map_location=DEVICE
        )
        ["model_state_dict"]

    )



    model.eval()


    pred=[]

    true=[]

    prob=[]



    with torch.no_grad():


        for images,targets in val_loader:


            images = images.to(
                DEVICE
            )


            outputs = model(
                images
            )


            probability = torch.softmax(
                outputs,
                dim=1
            )


            _,p = torch.max(
                outputs,
                1
            )


            pred.extend(
                p.cpu().numpy()
            )


            true.extend(
                targets.numpy()
            )


            prob.extend(
                probability.cpu().numpy()
            )



    true=np.array(true)

    pred=np.array(pred)

    prob=np.array(prob)



    auc = roc_auc_score(
        label_binarize(
            true,
            classes=[0,1,2]
        ),
        prob,
        average="macro",
        multi_class="ovr"
    )



    result = {


        "Fold":
        fold+1,


        "Accuracy":
        accuracy_score(
            true,
            pred
        ),


        "Precision":
        precision_score(
            true,
            pred,
            average="macro",
            zero_division=0
        ),


        "Recall":
        recall_score(
            true,
            pred,
            average="macro",
            zero_division=0
        ),


        "F1-score":
        f1_score(
            true,
            pred,
            average="macro",
            zero_division=0
        ),


        "AUC-ROC":
        auc,


        "Best Epoch":
        best_epoch


    }



    print(result)



    all_fold_results.append(
        result
    )



    # save fold metric


    with open(
        os.path.join(
            fold_path,
            "metrics.json"
        ),
        "w"
    ) as f:


        json.dump(
            result,
            f,
            indent=4
        )





# ============================================================
# Save all fold results
# ============================================================


import pandas as pd


df = pd.DataFrame(
    all_fold_results
)


df.to_csv(
    os.path.join(
        RESULT_PATH,
        "VGG16_COVID_5Fold_results.csv"
    ),
    index=False
)



# Save config


config = {


"Model":"VGG16",

"Dataset":
"COVID-19 Radiography Dataset",

"Classes":
CLASS_NAMES,


"Epoch":
EPOCHS,


"Batch_size":
BATCH_SIZE,


"Gradient_accumulation":
ACCUMULATION_STEP,


"Effective_batch":
BATCH_SIZE*ACCUMULATION_STEP,


"Learning_rate":
LR,


"Optimizer":
"Adam",


"Loss":
"CrossEntropyLoss with class weight",


"Image_size":
"224x224",


"Normalization":
"ImageNet",


"Fold":
5,


"Freeze":
"VGG16 block1-block3",


"Fine_tune":
"block4-block5 + classifier"

}



with open(
    os.path.join(
        RESULT_PATH,
        "config.json"
    ),
    "w"
) as f:


    json.dump(
        config,
        f,
        indent=4
    )



print("\nFinished 5 Fold Training")
print(
    "Results saved at:",
    RESULT_PATH
)