import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import os
import seaborn as sns

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register the hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(model_output)
        one_hot[0][target_class] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.detach().cpu()
        features = self.features.detach().cpu()
        
        weights = torch.mean(gradients, dim=(2, 3))[0, :]
        
        # Generate CAM
        cam = torch.zeros(features.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * features[0, i, :, :]
        
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.numpy(), target_class.item()

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image with mean and standard deviation."""
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_and_explain(model, dataloader, device, num_epochs, training_history=None, save_dir='./visualizations/'):
    """
    Generate comprehensive visualizations including training history and GradCAM
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()  # Set model to evaluation mode
    
    # 1. Plot Training History
    if training_history:
        plt.figure(figsize=(15, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), training_history['train_loss'], 'b-', label='Training Loss')
        plt.plot(range(1, num_epochs + 1), training_history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), training_history['train_accuracy'], 'b-', label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), training_history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. GradCAM Visualization
    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    
    try:
        # Get a batch of images
        batch = next(iter(dataloader))
        if len(batch) == 2:  # Training/validation data
            images, labels = batch
        else:  # Test data
            images = batch
            labels = None
        
        # Handle both single and dual image cases
        if not isinstance(images, list):
            images = images.to(device)
        else:
            images = [img.to(device) for img in images]
        
        dr_levels = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        
        # Process up to 5 images
        for idx in range(min(5, len(images) if not isinstance(images, list) else len(images[0]))):
            # Get current image
            if isinstance(images, list):
                image = images[0][idx:idx+1]
            else:
                image = images[idx:idx+1]
            
            # Generate CAM
            cam, predicted_class = gradcam.generate_cam(image)
            
            # Get true label if available
            if labels is not None:
                true_label = labels[idx].item() if not isinstance(labels, list) else labels[0][idx].item()
                true_label_text = f'\nTrue: {dr_levels[true_label]}'
            else:
                true_label_text = ''
            
            # Prepare visualization
            orig_img = denormalize_image(image[0])
            cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam_img = 0.7 * orig_img + 0.3 * heatmap
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img)
            plt.title(f'Original Image{true_label_text}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title(f'GradCAM Heatmap\nPredicted: {dr_levels[predicted_class]}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cam_img)
            plt.title('Combined Visualization')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/gradcam_visualization_{idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    finally:
        gradcam.remove_hooks()
    
    print(f"Visualizations saved to {save_dir}")