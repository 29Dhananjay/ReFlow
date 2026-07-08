import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

# Builder for each supported model. Every entry contains BatchNorm2d layers
# (required for the BN activation-statistics analysis) and loads pretrained
# ImageNet weights at the loader's 224x224 input size. This dict is the single
# source of truth for supported model names (see SUPPORTED_MODELS below).
MODEL_BUILDERS = {
    # ResNet / ResNeXt / Wide-ResNet
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'ResNet50': models.resnet50,
    'ResNet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_64x4d': models.resnext101_64x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    # VGG (BatchNorm variants only)
    'vgg11_bn': models.vgg11_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    # DenseNet
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    # RegNet
    'regnet_x_400mf': models.regnet_x_400mf,
    'regnet_x_3_2gf': models.regnet_x_3_2gf,
    'regnet_x_8gf': models.regnet_x_8gf,
    'regnet_x_32gf': models.regnet_x_32gf,
    'regnet_y_16gf': models.regnet_y_16gf,
    'regnet_y_128gf': models.regnet_y_128gf,  # SWAG weights (DEFAULT), no IMAGENET1K_V1
    # EfficientNet
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    # MobileNet
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    # Other compact / classic BN CNNs
    'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
    'mnasnet1_0': models.mnasnet1_0,
    'googlenet': models.googlenet,
}

# Names that have no IMAGENET1K_V1 weights; use their .DEFAULT (best available).
_DEFAULT_WEIGHT_MODELS = {"regnet_y_128gf"}

SUPPORTED_MODELS = list(MODEL_BUILDERS)


def get_model(model_name):
    """
    Load a pre-trained model by name (see SUPPORTED_MODELS / MODEL_BUILDERS).
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Model {model_name} not supported")

    if model_name in _DEFAULT_WEIGHT_MODELS:
        weights = "DEFAULT"
    else:
        weights = "IMAGENET1K_V1"

    return MODEL_BUILDERS[model_name](weights=weights)


def get_imagenet_loaders(data_path, batch_size, num_workers):
    """
    Returns ImageNet data loaders for training and validation.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = datasets.ImageFolder(root=f'{data_path}/train', transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.ImageFolder(root=f'{data_path}/val', transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader



def get_prunable_layers(model):
    """
    Returns a list of all prunable layers.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layers.append((name, module))
    return layers

def global_pruning(model, prune_amount):
    """
    Perform global magnitude pruning.
    """
    prunable_layers = get_prunable_layers(model)
    prune.global_unstructured(
        [(layer, 'weight') for _, layer in prunable_layers], 
        pruning_method=prune.L1Unstructured, 
        amount=prune_amount
    )
    return model


def layerwise_pruning(model, prune_amount):
    """
    Perform layer-wise pruning.
    """
    for _, layer in get_prunable_layers(model):
        prune.l1_unstructured(layer, name='weight', amount=prune_amount)
    return model


def sparsity(input_model):
    total_connections = 0
    unpruned_connections = 0
    
    # Get all prunable layers recursively
    prunable_layers = get_prunable_layers(input_model)
    
    for _, layer in prunable_layers:
        # Count weights
        total_connections += torch.numel(layer.weight.data)
        unpruned_connections += torch.count_nonzero(layer.weight.data)
        
        # Count bias if it exists
        if hasattr(layer, 'bias') and layer.bias is not None:
            total_connections += torch.numel(layer.bias.data)
            unpruned_connections += torch.count_nonzero(layer.bias.data)
    
    pruned_connections = total_connections - unpruned_connections
    spar = (pruned_connections / total_connections).double()
    return spar.item()



import torch

def print_layerwise_sparsity(model):
    """
    Prints parameter statistics and sparsity for each prunable layer,
    along with the total overall sparsity of the model.
    """
    prunable_layers = get_prunable_layers(model)
    
    total_net_params = 0
    total_net_active = 0
    
    # Define and print the table header
    header = f"{'Layer Name':<50} | {'Total':<10} | {'Active':<10} | {'Inactive':<10} | {'Sparsity (%)':<12}"
    print(header)
    print("-" * len(header))
    
    for name, layer in prunable_layers:
        # Tally weights
        layer_total = torch.numel(layer.weight.data)
        layer_active = torch.count_nonzero(layer.weight.data).item()
        
        # Tally biases if they exist
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer_total += torch.numel(layer.bias.data)
            layer_active += torch.count_nonzero(layer.bias.data).item()
            
        layer_inactive = layer_total - layer_active
        layer_sparsity = (layer_inactive / layer_total) * 100 if layer_total > 0 else 0.0
        
        # Accumulate network totals
        total_net_params += layer_total
        total_net_active += layer_active
        
        # Print row for the current layer
        print(f"{name:<50} | {layer_total:<10} | {layer_active:<10} | {layer_inactive:<10} | {layer_sparsity:>11.2f}%")
        
    print("-" * len(header))
    
    # Calculate and print overall network totals
    total_net_inactive = total_net_params - total_net_active
    total_sparsity = (total_net_inactive / total_net_params) * 100 if total_net_params > 0 else 0.0
    
    print(f"{'OVERALL TOTAL':<50} | {total_net_params:<10} | {total_net_active:<10} | {total_net_inactive:<10} | {total_sparsity:>11.2f}%")
    
    
def test(model, test_loader, device, max_batches=None):
    """
    Evaluate model accuracy.

    If ``max_batches`` is given, evaluation stops after that many batches,
    which is useful for quick smoke tests. ``None`` (default) evaluates the
    full loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100*correct / total

def model_update_bn(model, train_loader, device, count=100):
    """
    Fine-tune BatchNorm statistics.
    """
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            model(images)
            if i >= count:
                break
            
            
            
