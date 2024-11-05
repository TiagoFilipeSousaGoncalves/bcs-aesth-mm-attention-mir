import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn
from PIL import Image
from transformers import DeiTImageProcessor, DeiTModel, BeitImageProcessor, BeitModel 
from torchvision.models import resnet50, ResNet50_Weights



class DeiT_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's embeddings

class Beit_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]


class ResNet50_Base_224 (nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained ResNet50 model
        if weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained ResNet50 model
        base_model = resnet50(weights=weights)
        
        # Remove the final fully connected layer to use the model as a fixed feature extractor
        # Here we keep all layers up to, but not including, the final fully connected layer.
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer (fc layer)
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 2048, 1, 1) from the average pooling layer
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 2048) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to convert the output to shape (batch_size, 2048)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        
        return x
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            # Apply the necessary transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return transform(image)
        return transform
