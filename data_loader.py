import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DATA_PATH, IMG_SIZE, BATCH_SIZE

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_data_loaders():
    """Creates training and validation data loaders."""
    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_PATH, x), data_transforms[x])
        for x in ['train', 'validation']
    }
    
    # Create data loaders
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'validation']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes
    
    print(f"Class names: {class_names}")
    print(f"Training data size: {dataset_sizes['train']}")
    print(f"Validation data size: {dataset_sizes['validation']}")

    return dataloaders, class_names, dataset_sizes
