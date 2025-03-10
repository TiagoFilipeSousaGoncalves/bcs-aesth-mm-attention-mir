# Imports
from PIL import Image
import timm

# PyTorch Imports
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights

# Transformers Imports
from transformers import (
    AutoImageProcessor,
    ViTImageProcessor, 
    ViTModel, 
    DeiTImageProcessor, 
    DeiTModel, 
    BeitImageProcessor, 
    BeitModel, 
    Dinov2Model
)



# Class: Google_Base_Patch16_224
class Google_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained deit_tiny_patch16_224 ViT model
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            # pixel_values = processed['pixel_values']
            # pixel_values = pixel_values.permute(0, 2, 3, 1)
            # return pixel_values
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        # Assuming the model outputs the last_hidden_state directly
        featureVec = outputs.last_hidden_state[:, 0, :]  # Use outputs.last_hidden_state if no pooling
        return featureVec



# Class: DeiT_Base_Patch16_224
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



# Class: Beit_Base_Patch16_224
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



# Class: DinoV2_Base_Patch16_224
class DinoV2_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]



# Class: ResNet50_Base_224
class ResNet50_Base_224(nn.Module):
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



# Class: VGG16_Base_224
class VGG16_Base_224(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained VGG16 model
        if weights is not None:
            weights = VGG16_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained VGG16 model
        base_model = vgg16(weights=weights)
        
        # Remove the classifier layer to use the model as a fixed feature extractor
        # Here we keep all layers up to, but not including, the classifier layer.
        self.features = base_model.features  # Keep the convolutional feature extractor part
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 512, 7, 7)
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 512) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to resize the output to shape (batch_size, 512)
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



# Class: VGG16_Base_224_MLP
class VGG16_Base_224_MLP(nn.Module):
    def __init__(self, weights=VGG16_Weights.DEFAULT, feature_dim=512, embedding_size=256):
        super().__init__()
        # Load the pre-trained VGG16 model
        if weights is not None:
            weights = VGG16_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained VGG16 model
        base_model = vgg16(weights=weights)
        
        # Remove the classifier layer to use the model as a fixed feature extractor
        # Here we keep all layers up to, but not including, the classifier layer.
        self.features = base_model.features  # Keep the convolutional feature extractor part
        
        # The output of 'self.features' will be a tensor of shape (batch_size, 512, 7, 7)
        # We will add an AdaptiveAvgPool layer to convert it to (batch_size, 512) which is easier to use in most tasks
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Additional MLP Layers
        self.fc1 = nn.Linear(feature_dim * 49, embedding_size)  # 512*7*7 = 25088 inputs to 256
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size, 128)  # Second MLP layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)  # Third MLP layer

    def forward(self, x):
        # Pass input through the feature layers
        x = self.features(x)
        
        # Apply adaptive pooling to resize the output to shape (batch_size, 512, 7, 7)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Pass through the fully connected layers with ReLU activation
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
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



# Class: Google_Base_Patch16_224_MLP
class Google_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(Google_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained deit_tiny_patch16_224 ViT model
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer (change 768 to your feature size)
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        # Assuming the model outputs the last_hidden_state directly
        featureVec = outputs.last_hidden_state[:, 0, :]  # Use outputs.last_hidden_state if no pooling
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: ResNet50_Base_224_MLP
class ResNet50_Base_224_MLP(nn.Module):
    def __init__(self, feature_dim=2048, embedding_size=512, weights=ResNet50_Weights.DEFAULT):
        super().__init__()
        # Load the pre-trained ResNet50 model
        if weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V1  # Use the default ImageNet weights

        # Load the pre-trained ResNet50 model
        base_model = resnet50(weights=weights)
                
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Adaptive pooling to make sure output size is consistent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer to transform the feature space
        self.fc1 = nn.Linear(feature_dim, embedding_size)
        
        # Add another layer, if needed, you can increase the complexity here
        self.fc2 = nn.Linear(embedding_size, 256)

        # Optional: Add a batch normalization layer
        self.batch_norm = nn.BatchNorm1d(256)

        # Optional: Add a Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features from the base model
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Pass through the first fully connected layer
        x = F.relu(self.fc1(x))

        # Pass through the second fully connected layer (with optional batch normalization and dropout)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = F.dropout(x, p=0.5, training=self.training)

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



# Class: DeiT_Base_Patch16_224_MLP
class DeiT_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(DeiT_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained DEIT model
        self.feature_extractor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-patch16-224')
    
        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # Adjust the input size to match the output size of the last hidden layer of DeiT
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Further MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's embeddings
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: DinoV2_Base_Patch16_224_MLP
class DinoV2_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(DinoV2_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained DinoV2 model
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')

        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer; adjust the size to match DinoV2 output
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings from the [CLS] token, which is typically used for classification tasks
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: Beit_Base_Patch16_224_MLP
class Beit_Base_Patch16_224_MLP(nn.Module):
    def __init__(self):
        super(Beit_Base_Patch16_224_MLP, self).__init__()
        # Load the pre-trained Beit model
        self.feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

        # Define MLP layers
        self.fc1 = nn.Linear(768, 512)  # First MLP layer; size must match the output feature dimension of Beit
        self.relu1 = nn.ReLU()          # ReLU activation
        self.fc2 = nn.Linear(512, 256)  # Second MLP layer
        self.relu2 = nn.ReLU()          # ReLU activation

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        featureVec = outputs.last_hidden_state[:, 0, :]  # Extract the embeddings from the [CLS] token or equivalent
        x = self.fc1(featureVec)
        x = self.relu1(x)
        x = self.fc2(x)
        featureVec = self.relu2(x)
        return featureVec



# Class: CrossViT_Tiny240
class CrossViT_Tiny240(nn.Module):

    # Method: __init__
    def __init__(self):
        super(CrossViT_Tiny240, self).__init__()
        model = timm.create_model(
            'crossvit_tiny_240.in1k', 
            pretrained=True, 
            num_classes=0
        )
        self.model = model
        return
    
    def get_transform(self):
        def transform(image_path):
            # self.model.eval()
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: LeViTConv256
class LeViTConv256(nn.Module):

    # Method: __init__
    def __init__(self):
        super(LeViTConv256, self).__init__()
        model = timm.create_model(
            'levit_conv_256.fb_dist_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: ConViT_Tiny
class ConViT_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(ConViT_Tiny, self).__init__()
        
        model = timm.create_model(
            'convit_tiny.fb_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model

        return


    # Method: get_transform
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform


    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class: MaxViT_Tiny_224
class MaxViT_Tiny_224(nn.Module):

    # Method: __init__
    def __init__(self):
        super(MaxViT_Tiny_224, self).__init__()
        model = timm.create_model(
            'maxvit_tiny_tf_224.in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class MViTv2_Tiny
class MViTv2_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(MViTv2_Tiny, self).__init__()
        model = timm.create_model(
            'mvitv2_tiny.fb_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Class DaViT_Tiny
class DaViT_Tiny(nn.Module):

    # Method: __init__
    def __init__(self):
        super(DaViT_Tiny, self).__init__()
        model = timm.create_model(
            'davit_tiny.msft_in1k',
            pretrained=True,
            num_classes=0
        )
        self.model = model
        return
    

    # Method: get_transforms
    def get_transform(self):
        def transform(image_path):
            data_config = timm.data.resolve_model_data_config(self.model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            image = Image.open(image_path).convert('RGB')
            image_trans = transforms(image)
            return image_trans
        return transform
    

    # Method: forward
    def forward(self, input):
        featureVec = self.model(input)
        return featureVec



# Dictionary: Models dictionary
MODELS_DICT = {

    # Image
    "Google_Base_Patch16_224":Google_Base_Patch16_224(),
    "DeiT_Base_Patch16_224":DeiT_Base_Patch16_224(),
    "Beit_Base_Patch16_224":Beit_Base_Patch16_224(),
    "DinoV2_Base_Patch16_224":DinoV2_Base_Patch16_224(),
    "ResNet50_Base_224":ResNet50_Base_224(),
    "VGG16_Base_224":VGG16_Base_224(),
    "CrossViT_Tiny240":CrossViT_Tiny240(),
    "LeViTConv256":LeViTConv256(),
    "ConViT_Tiny":ConViT_Tiny(),
    "MaxViT_Tiny_224":MaxViT_Tiny_224(),
    "MViTv2_Tiny":MViTv2_Tiny(),
    "DaViT_Tiny":DaViT_Tiny(),

    # Multimodal
    "Google_Base_Patch16_224_MLP":Google_Base_Patch16_224_MLP(),
    "DinoV2_Base_Patch16_224_MLP":DinoV2_Base_Patch16_224_MLP(),
    "Beit_Base_Patch16_224_MLP":Beit_Base_Patch16_224_MLP(),
    "DeiT_Base_Patch16_224_MLP":DeiT_Base_Patch16_224_MLP(),
    "ResNet50_Base_224_MLP":ResNet50_Base_224_MLP(),
    "VGG16_Base_224_MLP":VGG16_Base_224_MLP()
}