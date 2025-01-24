import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

# بارگذاری داده‌ها از فایل اکسل
df = pd.read_excel("output_combined_data.xlsx")

# دیکشنری برای تبدیل برچسب‌های متنی به عددی
label_mapping = {label: idx for idx, label in enumerate(sorted(set(df['label_location'].values)))}

# چاپ تعداد کلاس‌ها و mapping
print(f"Number of unique labels: {len(label_mapping)}")
print("Label mapping:", label_mapping)

# Custom Dataset برای PyTorch
class CustomDataset(Dataset):
    def __init__(self, matrices, labels, transform=None):
        self.matrices = matrices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        label = self.labels[idx]

        if matrix is not None:
            rows = matrix.shape[0]
            matrix = matrix.reshape(rows, 90, 3)

        if self.transform and matrix is not None:
            matrix = self.transform(matrix)

        return matrix, label


# تعریف مدل ResNet به همراه classifier سفارشی
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # حذف یک بعد اضافی
        return self.resnet(x)


# تابع آماده‌سازی DataLoaderها
def prepare_dataloader(df, wifi_band_filter, env_filter, batch_size, transform=None):
    filtered_df = df[(df['wifi_band'] == wifi_band_filter) & (df['envs_numeric'] == env_filter)]
    mat_names = filtered_df['mat_name'].values
    label_location = filtered_df['label_location'].values

    matrices = []
    labels = []
    for mat_name, label in zip(mat_names, label_location):
        matrix_file = f"F:/WiFi-based Multi-user Activity Sensing/DataSet/amp/{mat_name}.npy"
        try:
            matrix = np.load(matrix_file)
            matrices.append(matrix)
            labels.append(label_mapping[label])
        except FileNotFoundError:
            print(f"File {matrix_file} not found.")

    dataset = CustomDataset(matrices, labels, transform=transform)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, valid_loader


# تابع آموزش مدل
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, device="cuda"):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):  # اضافه کردن شمارنده batch
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            inputs = inputs.unsqueeze(1)  # اضافه کردن بعد کانال

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # محاسبه دقت هر بچ
            batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
            print(f"Batch {batch_idx}/{len(train_loader)}, Batch Accuracy: {batch_accuracy:.2f}%")

        # دقت کل اپوک برای داده‌های آموزش
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # اعتبارسنجی در پایان هر اپوک
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid
        print(f"Validation Accuracy: {valid_accuracy:.2f}%")



# تابع collate سفارشی برای پد کردن ماتریس‌ها
def custom_collate_fn(batch):
    matrices, labels = zip(*batch)
    max_seq_len = max(matrix.shape[1] for matrix in matrices)
    padded_matrices = [
        F.pad(torch.tensor(matrix).clone().detach(), (0, 0, 0, max_seq_len - matrix.shape[1])) for matrix in matrices
    ]
    labels = torch.tensor(labels, dtype=torch.long)
    return torch.stack(padded_matrices), labels


# تعریف پارامترها
wifi_band_filter = 2.4
env_filter = 1
batch_size = 128
num_classes = len(label_mapping)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# آماده‌سازی DataLoaderها
train_loader, valid_loader = prepare_dataloader(df, wifi_band_filter, env_filter, batch_size, transform)

# تعریف مدل، تابع هزینه و بهینه‌ساز
model = ResNetClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50, device='cuda')
