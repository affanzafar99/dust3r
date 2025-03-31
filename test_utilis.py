from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.datasets.freiburgDataset import freiburgDataset 
import tqdm
import matplotlib.pyplot as plt
import numpy as np 
import random

import torchvision.transforms as T
from PIL import Image
import torch
def compute_accuracy(pred1, gt1):        
# Compute the L2 loss for depth (between predicted and ground truth depth)
        depth_pred = pred1['pts3d'][..., 2]  # Extract the predicted depth (z-coordinate)
        gt_depthmap = torch.tensor(gt1["depthmap"]).to(depth_pred.device)
        # l1 = self.criterion(depth_pred,gt_depthmap)
        # depth_loss = torch.sqrt(torch.mean((depth_pred - gt_depthmap)**2))

        
        epsilon = 1e-8  # To avoid log(0)
        log_diff = torch.log(depth_pred + epsilon) - torch.log(gt_depthmap + epsilon)
        loss = torch.sqrt(torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2))

        # # Compute accuracy-based loss (Threshold-based accuracy)
        threshold = torch.max(depth_pred / gt_depthmap, gt_depthmap / depth_pred)
        acc_1_25 = torch.mean((threshold < 1.25).float())  # Accuracy with threshold 1.25
        acc_1_25_2 = torch.mean((threshold < 1.25**2).float())  # Accuracy with threshold 1.25^2



        # # Track individual losses for debugging or analysis
        details = {
            "RMSE Loss": loss.item(),
            "Accuracy < 1.25": acc_1_25.item(),
            "Accuracy < 1.25^2": acc_1_25_2.item()
        }
        
        return details

def average_metrics(array):
    if not array:
        print("Empty array. No averages to compute.")
        return

    keys = array[0].keys()
    averages = {key: sum(d[key] for d in array) / len(array) for key in keys}
    
    for key, value in averages.items():
        print(f"{key}: {value:.4f}")
        
def view_depth(view11,pred1):
    pts3d = pred1['pts3d']
    # If shape is [B, H, W, 3], permute to [B, 3, H, W]
    if pts3d.shape[-1] == 3:
        pts3d = pts3d.permute(0, 3, 1, 2)

    # Extract the Z-coordinate (depth)
    depth_pred = pts3d[0, 2, :, :].squeeze()


    img = view11["img"]
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    depth_gt = view11["depthmap"]
    img = view11["img"][0]

    if depth_pred.max() <= 1.0:
        depth_pred = (depth_pred * 255).astype('uint8')
    # Create a subplot with 1 row and 3 columns
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))

    # Plot the RGB image (img)
    ax[0].imshow(img)  # For RGB, no cmap needed
    ax[0].axis('off')  # Hide axes for a cleaner view
    ax[0].set_title('RGB Image')

    # Plot the predicted depth map (depth_pred) with colormap 'turbo'
    ax[1].imshow(depth_pred, cmap='inferno') 
    ax[1].axis('off')  # Hide axes
    ax[1].set_title('Predicted Depth')

    # Plot the ground truth depth map (depth_gt) 
    ax[2].imshow(depth_gt, cmap='inferno')  
    ax[2].axis('off')  # Hide axes
    ax[2].set_title('Ground Truth Depth')

    # Show the images
    plt.show()

def test_model(model,test_dataset):
    test_dataset =freiburgDataset(ROOT='/home/user/elwakeely1/DataParam', split='Test', resolution=224, aug_crop=16,method = 'RANSAC')
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    accuracy_list = []
    random_indices = random.sample(range(len(test_dataset)), 5)

    for i in (range(len(test_dataset))):
        view11,view21 = test_dataset[i]
        img = view11['img']
        images = load_images([img,img], size=224,train = False)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs,model, None, batch_size=batch_size,verbose = False)

        # at this stage, you have the raw dust3r predictions
        view12, pred1 = output['view1'], output['pred1']
        view22, pred2 = output['view2'], output['pred2']
        pts3d = pred1['pts3d']
        # If shape is [B, H, W, 3], permute to [B, 3, H, W]
        if pts3d.shape[-1] == 3:
            pts3d = pts3d.permute(0, 3, 1, 2)

        # Extract the Z-coordinate (depth)
        depth_pred = pts3d[0, 2, :, :].squeeze()


        img = view11["img"]
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
            
        if i in random_indices:
            depth_gt = view11["depthmap"]
            view_depth(view11,pred1)
        
        img = view11["img"][0]

        if depth_pred.max() <= 1.0:
            depth_pred = (depth_pred * 255).astype('uint8')
        accuracy_list.append(compute_accuracy(pred1,view11))
    return accuracy_list