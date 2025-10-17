import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from transformers import ViTConfig, ViTForImageClassification
#from dataset import DrowsinessDataset
#%run dataset.py
import sys
import os
sys.path.append(os.getcwd())
from dataset import DrowsinessDataset
import multiprocessing
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau

def format_time(seconds):
    """
    Converts seconds into a readable time format (HH:MM:SS.ss).
    Returns a formatted string representing the time.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def load_data(data_dir):
    """
    Loads image data from the specified directory.
    Returns lists of image paths and corresponding labels.
    """
    image_paths = []
    labels = []
    for label, folder in enumerate(['not_drowsy', 'drowsy']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label)
    return image_paths, labels

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    Trains the model using the provided data loaders.
    Performs validation after each epoch and logs the results.
    Returns the trained model.
    """
    start_time = time.time()
    print(f"Using device: {device}")

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    if torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None


    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        batch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(images).logits
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:  # Changed from 10 to 20
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Batch Time: {format_time(batch_time)}')
                batch_start_time = time.time()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_start_time = time.time()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_time = time.time() - val_start_time
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%, '
              f'Epoch Time: {format_time(epoch_time)}, '
              f'Validation Time: {format_time(val_time)}')

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.savefig('training_curves.png')
    plt.close()

    total_time = time.time() - start_time
    print(f"Training completed in {format_time(total_time)}. Curves saved as 'training_curves.png'")
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluates the trained model on the test dataset.
    Computes and displays accuracy, classification report, and confusion matrix.
    """
    start_time = time.time()
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Not Drowsy', 'Drowsy'])
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Drowsy', 'Drowsy'], rotation=45)
    plt.yticks(tick_marks, ['Not Drowsy', 'Drowsy'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.savefig('confusion_matrix.png')
    plt.close()

    eval_time = time.time() - start_time
    print(f"Evaluation completed in {format_time(eval_time)}. Confusion matrix saved as 'confusion_matrix.png'")

def main():
    """
    Main function to execute the drowsiness detection model training and evaluation.
    Loads data, prepares datasets, trains the model, and evaluates its performance.
    """
    overall_start_time = time.time()
    torch.backends.cudnn.benchmark = True
    mp.set_sharing_strategy('file_system')
    ctx = mp.get_context('spawn')

    # Check for MPS (Metal) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data loading and preprocessing
    data_loading_start = time.time()
    data_dir = "/Users/kaviyarasisankarapandiyan/Documents/MscDataScience/Semester3/DataScienceProject/CodingPart/Dataset"
    image_paths, labels = load_data(data_dir)

    print(f"Total number of samples: {len(image_paths)}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of test samples: {len(X_test)}")

    train_drowsy = sum(y_train)
    train_not_drowsy = len(y_train) - train_drowsy
    test_drowsy = sum(y_test)
    test_not_drowsy = len(y_test) - test_drowsy

    print(f"\nTraining set class distribution:")
    print(f"  Drowsy: {train_drowsy} ({train_drowsy/len(y_train)*100:.2f}%)")
    print(f"  Not Drowsy: {train_not_drowsy} ({train_not_drowsy/len(y_train)*100:.2f}%)")

    print(f"\nTest set class distribution:")
    print(f"  Drowsy: {test_drowsy} ({test_drowsy/len(y_test)*100:.2f}%)")
    print(f"  Not Drowsy: {test_not_drowsy} ({test_not_drowsy/len(y_test)*100:.2f}%)")

    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Changed (224, 224)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = DrowsinessDataset(X_train, y_train, transform=transform)
    val_dataset = DrowsinessDataset(X_val, y_val, transform=transform)
    test_dataset = DrowsinessDataset(X_test, y_test, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2, persistent_workers=True, multiprocessing_context=ctx) # Changes batch_size=32
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=2, persistent_workers=True, multiprocessing_context=ctx) # num_workers=4
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=2, persistent_workers=True, multiprocessing_context=ctx)

    data_loading_time = time.time() - data_loading_start
    print(f"\nData loading and preprocessing completed in {format_time(data_loading_time)}")

    print("\nLoading pre-trained ViT model...")
    model_loading_start = time.time()

    config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=2)
    config.image_size = 160  # Match this with your new image size
    model = ViTForImageClassification(config)

    #     model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2, ignore_mismatched_sizes=True)
    model_loading_time = time.time() - model_loading_start
    print(f"Model loaded successfully in {format_time(model_loading_time)}")

    print("\nStarting model training...")
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=10)

    print("\nEvaluating model on test set...")
    evaluate_model(trained_model, test_loader, device)

    overall_time = time.time() - overall_start_time
    print(f"\nEntire process completed in {format_time(overall_time)}")

if __name__ == "__main__":
    #     multiprocessing.set_start_method('spawn', force=True)
    main()
    