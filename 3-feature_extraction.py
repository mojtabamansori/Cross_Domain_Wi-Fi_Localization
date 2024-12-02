import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

# بارگذاری داده‌ها از اکسل
df = pd.read_excel("output_combined_data.xlsx")

# دیکشنری برای تبدیل برچسب‌های متنی به عددی
label_mapping = {label: idx for idx, label in enumerate(sorted(set(df['label_location'].values)))}

# چاپ تعداد کلاس‌ها و label_mapping
print(f"Number of unique labels: {len(label_mapping)}")  # باید ۵ باشد
print("Label mapping:", label_mapping)

# Data loader function
def data_loader(df, wifi_band_filter, env_filter, batch_size=32):
    filtered_df = df[(df['wifi_band'] == wifi_band_filter) & (df['envs_numeric'] == env_filter)]
    mat_names = filtered_df['mat_name'].values
    user_name = filtered_df['user_name'].values
    label_location = filtered_df['label_location'].values

    matrices = []
    labels = []
    for mat_name, label in zip(mat_names, label_location):
        matrix_file = f"F:/WiFi-based Multi-user Activity Sensing/DataSet/amp/{mat_name}.npy"
        try:
            matrix = np.load(matrix_file)
            matrices.append(matrix)
            labels.append(label_mapping[label])  # تبدیل برچسب‌ها به عددی
        except FileNotFoundError:
            print(f"File {matrix_file} not found.")
            matrices.append(None)
            labels.append(None)

    num_batches = len(matrices) // batch_size + (1 if len(matrices) % batch_size != 0 else 0)

    for i in range(num_batches):
        batch_matrices = matrices[i * batch_size: (i + 1) * batch_size]
        batch_user_name = user_name[i * batch_size: (i + 1) * batch_size]
        batch_label_location = labels[i * batch_size: (i + 1) * batch_size]  # استفاده از برچسب‌های عددی
        yield batch_matrices, batch_user_name, batch_label_location


# Custom Dataset for PyTorch
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
            # تغییر شکل ماتریس از (ردیف, 3, 3, 30) به (ردیف, 3, 90, 3)
            rows = matrix.shape[0]
            matrix = matrix.reshape(rows, 90, 3)  # 3 کانال، 30 ارتفاع، 3 عرض

        if self.transform and matrix is not None:
            matrix = self.transform(matrix)

        return matrix, label


# Define ResNet model with custom classifier
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        # افزودن بعد کانال به ورودی (کانال‌ها، ارتفاع، عرض)
        x = x.squeeze(1)  # از ۵ بعد به ۴ بعد تبدیل می‌شود
        return self.resnet(x)


# Training function
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=2):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.float()
            inputs = inputs.unsqueeze(1)  # اضافه کردن بعد کانال
            labels = labels.long()  # تبدیل labels به نوع long

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # محاسبه دقت برای هر بچ
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct_train / total_train
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # محاسبه دقت برای داده‌های اعتبارسنجی
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.float()
                inputs = inputs.unsqueeze(1)  # اضافه کردن بعد کانال
                labels = labels.long()

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        valid_accuracy = 100 * correct_valid / total_valid
        print(f'Validation Accuracy: {valid_accuracy:.2f}%')


# Custom collate function for padding
def custom_collate_fn(batch):
    matrices, labels = zip(*batch)
    max_seq_len = max(matrix.shape[1] for matrix in matrices)
    padded_matrices = [F.pad(torch.tensor(matrix).clone().detach(), (0, 0, 0, max_seq_len - matrix.shape[1])) for matrix
                       in matrices]

    # تبدیل labels به torch.tensor از نوع long
    labels = torch.tensor(labels, dtype=torch.long)

    return torch.stack(padded_matrices), labels


# پارامترهای ورودی
wifi_band_filter = 2.4
env_filter = 1
batch_size = 64

# Process each batch
for batch_matrices, batch_user_name, batch_label_location in data_loader(df, wifi_band_filter, env_filter, batch_size):
    valid_data = [(mat, label) for mat, label in zip(batch_matrices, batch_label_location) if mat is not None]
    if not valid_data:
        continue

    matrices, labels = zip(*valid_data)

    # Transform and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = CustomDataset(matrices, labels, transform=transform)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Define model, loss function, and optimizer
    num_classes = 5  # تعداد کلاس‌ها برابر با ۵
    model = ResNetClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_model(model, train_loader, valid_loader, criterion, optimizer)
