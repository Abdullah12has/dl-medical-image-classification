# this file contains resnet 18 running pretrained weights + self attention + all layers unfrozen + Boosting

import copy
import os
import random
import sys

import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image, adjust_gamma
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
# Hyper Parameters
batch_size = 32
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 25


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img



transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    # FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomApply([transforms.Lambda(lambda img: adjust_gamma(img, gamma=1.5))], p=0.3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter

    def forward(self, x):
        batch_size, C, H, W = x.size()  # Input feature map dimensions: (B, C, H, W)

        # Query, Key, and Value transformations
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C//8)
        key = self.key_conv(x).view(batch_size, -1, H * W)  # Shape: (B, C//8, H*W)
        value = self.value_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)

        # Compute attention weights
        attention = torch.bmm(query, key)  # Shape: (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)  # Normalize attention weights across spatial dimensions

        # Weighted sum of values
        out = torch.bmm(attention, value).permute(0, 2, 1)  # Shape: (B, C, H*W)
        out = out.view(batch_size, C, H, W)  # Reshape back to spatial dimensions

        # Apply learnable scaling and residual connection
        out = self.gamma * out + x
        return out


class MyModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.52):
        super().__init__()

        self.backbone = models.resnet34(pretrained=True)
        # Get the input features for the classifier dynamically
        in_features = self.backbone.fc.in_features

        for param in self.backbone.parameters():
            param.requires_grad = True

        # Self-attention layer (applied to intermediate feature maps)
        self.self_attention = SelfAttention(in_channels=512)
        self.self_attention3 = SelfAttention(in_channels=256)
        self.self_attention4 = SelfAttention(in_channels=512)

        # Replace the classifier with a custom one
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract intermediate feature maps from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.self_attention3(x)
        x = self.backbone.layer4(x)
        x = self.self_attention4(x)

        # Apply self-attention to the feature maps
        # x = self.self_attention(x)

        # Apply global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (B, 512)

        # Pass through the classifier
        x = self.backbone.fc(x)
        return x




# class BaggingEnsemble(nn.Module):
#     def __init__(self, base_model, num_models, num_classes):
#         super(BaggingEnsemble, self).__init__()
#         self.models = nn.ModuleList([base_model for _ in range(num_models)])
#         self.num_models = num_models
#         self.num_classes = num_classes

#     def forward(self, x):
#         # Collect predictions from all models
#         outputs = torch.stack([model(x) for model in self.models], dim=0)  # Shape: (num_models, batch_size, num_classes)
#         # Average predictions across all models
#         ensemble_output = torch.mean(outputs, dim=0)  # Shape: (batch_size, num_classes)
#         return ensemble_output

class BoostingEnsemble(nn.Module):
    def __init__(self, models, num_classes=5):
        super(BoostingEnsemble, self).__init__()
        self.models = models  # List of models
        self.num_classes = num_classes
        self.boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    def forward(self, x):
        # Collect predictions from all models
        all_preds = []
        for model in self.models:
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                output = model(x)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds.cpu().numpy())
        
        # Convert to a format suitable for the boosting model
        all_preds = np.array(all_preds).T  # Shape: (batch_size, num_models)
        
        return all_preds

    def fit_boosting(self, X_train, y_train):
        # Train the boosting model
        self.boosting_model.fit(X_train, y_train)
    
    def predict_boosting(self, X_test):
        return self.boosting_model.predict(X_test)
    
    def evaluate_boosting(self, X_test, y_test):
        y_pred = self.predict_boosting(X_test)
        kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f'Boosting Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f}')
        return kappa, accuracy, precision, recall



def train_model_with_boosting(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25, checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0

    # Initialize training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    all_train_preds = []
    all_train_labels = []

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []

        model.train()
        epoch_train_preds = []
        epoch_train_labels = []
        
        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                epoch_train_preds.extend(preds.cpu().numpy())
                epoch_train_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())
                pbar.update(1)

        lr_scheduler.step()

        # Calculate epoch metrics
        epoch_loss = sum(running_loss) / len(running_loss)
        train_metrics = compute_metrics(epoch_train_preds, epoch_train_labels)
        train_kappa, train_accuracy, train_precision, train_recall = train_metrics

        # Calculate validation metrics and loss
        val_running_loss = []
        val_preds = []
        val_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels.long())
                
                preds = torch.argmax(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_running_loss.append(val_loss.item())

        val_epoch_loss = sum(val_running_loss) / len(val_running_loss)
        val_metrics = compute_metrics(val_preds, val_labels)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics

        # Store metrics in training history
        training_history['train_loss'].append(epoch_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_loss'].append(val_epoch_loss)
        training_history['val_accuracy'].append(val_accuracy)

        print(f'[Train] Kappa: {train_kappa:.4f} Accuracy: {train_accuracy:.4f} '
              f'Precision: {train_precision:.4f} Recall: {train_recall:.4f} Loss: {epoch_loss:.4f}')
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f} Loss: {val_epoch_loss:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

    # Save the best model state
    torch.save(best_model, checkpoint_path)
    print(f'Best model saved at epoch {best_epoch} with validation kappa: {best_val_kappa:.4f}')
    
    return model, training_history




def train_and_extract_features(model, train_loader, val_loader, device, criterion, optimizer, num_epochs=25):
    model.train()
    all_train_features, all_train_labels = [], []
    
    # Initialize training history dictionary
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Initialize the booster
    booster = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        epoch_features = []
        epoch_labels = []
        epoch_preds = []  # Track predictions for accuracy

        with tqdm(total=len(train_loader), desc='Training', unit='batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
                pbar.update(1)

                # Collect features and predictions
                with torch.no_grad():
                    features = outputs.cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    epoch_features.append(features)
                    epoch_labels.append(labels.cpu().numpy())
                    epoch_preds.extend(preds)

        # Calculate training metrics
        epoch_features = np.concatenate(epoch_features)
        epoch_labels = np.concatenate(epoch_labels).flatten()
        epoch_loss = sum(running_loss) / len(running_loss)
        train_accuracy = accuracy_score(epoch_labels, epoch_preds)
        
        # Store training metrics
        training_history['train_loss'].append(epoch_loss)
        training_history['train_accuracy'].append(train_accuracy)
        
        # Add to overall features and labels
        all_train_features.append(epoch_features)
        all_train_labels.append(epoch_labels)

        print(f'[Epoch {epoch}] Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # Validation phase
        model.eval()
        all_val_features, all_val_labels = [], []
        val_running_loss = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels.long())
                val_running_loss.append(val_loss.item())
                
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(predictions)
                all_val_features.append(outputs.cpu().numpy())
                all_val_labels.append(labels.cpu().numpy())

        val_features = np.concatenate(all_val_features)
        val_labels = np.concatenate(all_val_labels)
        val_epoch_loss = sum(val_running_loss) / len(val_running_loss)
        val_accuracy = accuracy_score(val_labels.flatten(), val_preds)
        
        # Store validation metrics
        training_history['val_loss'].append(val_epoch_loss)
        training_history['val_accuracy'].append(val_accuracy)

        # Train booster on accumulated features
        train_features_combined = np.concatenate(all_train_features)
        train_labels_combined = np.concatenate(all_train_labels)
        booster.fit(train_features_combined, train_labels_combined)
        
        # Evaluate boosting
        boost_preds = booster.predict(val_features)
        boost_accuracy = accuracy_score(val_labels.flatten(), boost_preds)
        print(f'[Epoch {epoch}] Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print(f'[Epoch {epoch}] Boosting Validation Accuracy: {boost_accuracy:.4f}')

    # Return final features, labels, and training history
    final_train_features = np.concatenate(all_train_features)
    final_train_labels = np.concatenate(all_train_labels)
    final_val_features = val_features
    final_val_labels = val_labels

    return final_train_features, final_train_labels, final_val_features, final_val_labels, training_history


if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    # This will affect the model definition, dataset pipeline, training and evaluation

 
    mode = 'single'  # forward single image to the model each time 

    assert mode in ('single', 'dual')

    # Define the model
    if mode == 'single':
        model = MyModel()
    # else:
    #     model = MyDualModel()

    print(model, '\n')
    print('Pipeline Mode:', mode)

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Use GPU device is possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    
    state_dict = torch.load('./pre/pretrained/resnet34.pth', map_location='cpu')
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"backbone.{key}"  # Prefix with 'backbone.'
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    
    

    # Move class weights to the device
    model = model.to(device)

    # Optimizer and Learning rate scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and evaluate the model with the training and validation set
    train_features, train_labels, val_features, val_labels, training_history = train_and_extract_features(
        model, train_loader, val_loader, device, criterion, optimizer, num_epochs=num_epochs
    )
    
    # Apply boosting ensemble method
    train_labels = train_labels.flatten()
    val_labels = val_labels.flatten()

    booster = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    booster.fit(train_features, train_labels)
    val_preds = booster.predict(val_features)

    # Evaluate
    accuracy = accuracy_score(val_labels, val_preds)
    print(f'Boosting Validation Accuracy: {accuracy:.4f}')

    # Generate visualizations
    print("\nGenerating visualizations...")
    from visualization import visualize_and_explain
    
    # Create visualization directory
    os.makedirs('./visualizations', exist_ok=True)
    
    visualize_and_explain(
        model=model,
        dataloader=val_loader,
        device=device,
        num_epochs=num_epochs,
        training_history=training_history,
        save_dir='./visualizations/'
    )

    # Make predictions on testing set
    evaluate_model(model, test_loader, device, test_only=True)
