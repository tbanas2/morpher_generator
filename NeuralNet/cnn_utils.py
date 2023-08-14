import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import requests
import tarfile

# CNN Utility Functions
# Code written by Luke Banaszak or Rick Suggs unless otherwise noted


# Adapted from https://github.com/nalbert9/Facial-Keypoint-Detection
def net_sample_output(model, dataloader, num_keypoints=58):
    """Iterates through results in the dataloader, and invokes the model on each sample.

    Args:
        model (torch.nn.Module): The Pytorch Model
        dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader
        num_keypoints (int, optional): Number of keypoints to reshape results into. Defaults to 58.

    Yields:
        tuple: tuple of (ndarray, ndarray, ndarray): (image, output keypoints, input keypoints)
    """

    # iterate through the test dataset
    for sample in dataloader:
        # get sample data: images and ground truth keypoints
        image = sample["image"]
        key_pts = sample["landmarks"]
        # convert images to FloatTensors
        image = (
            image.type(torch.cuda.FloatTensor)
            if torch.cuda.is_available()
            else image.type(torch.FloatTensor)
        )

        # forward pass to get net output
        output_pts = model(image)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], -1, num_keypoints)

        # break after first image is tested
        yield image, output_pts, key_pts

        break

# Adapted from https://github.com/nalbert9/Facial-Keypoint-Detection
def visualize_output(test_images, test_outputs, gt_pts=None):
    """
    Displays results from keypoint detection

    Args:
        test_images (ndarray): array of input images
        test_outputs (ndarray): array of detected keypoints
        gt_pts (ndarray, optional): array of original keypoints. Defaults to None.
    """    
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.1)

    for i, (test_image, test_output) in enumerate(zip(test_images, test_outputs)):
        # un-transform the image data
        image = test_image.data
        image = image.cpu().numpy()
        if image.min() < 0:
            # normalize image from -0.5 - 0.5 range back to 0 - 1 range for display
            image = image + 0.5
        image = np.transpose(image, (1, 2, 0))

        # un-transform the predicted key_pts data
        predicted_key_pts = test_output.data.cpu().numpy()
        # predicted_key_pts = predicted_key_pts*50.0+100

        grid[i].set_xticks([])
        grid[i].set_yticks([])

        if image.shape[2] == 1:
            grid[i].imshow(image, cmap="gray")
        elif image.shape[2] == 3:
            grid[i].imshow(image)

        grid[i].scatter(*predicted_key_pts, s=5, marker=".", c="m")
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i].cpu().numpy()
            # ground_truth_pts = ground_truth_pts*50.0+100
            grid[i].scatter(*ground_truth_pts, s=5, marker=".", c="g")

    plt.axis("off")
    plt.show()


# Inspired by https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
def train_and_validate(
    model, train_dataloader, optimizer, criterion, test_dataloader, n_epochs
):
    """
    Trains and validates the model

    Args:
        model (torch.nn.Module): The PyTorch model
        train_dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader for training data
        optimizer (torch.optim.Optimizer): The PyTorch Optimizer 
        criterion (torch.nn.MSELoss): The PyTorch Criterion
        test_dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader for validation data
        n_epochs (int): number of epochs to train

    Returns:
        tuple: (list[float], list[float]) A tuple of the training loss and validation loss statistics
    """    
    training_loss = []
    validation_loss = []

    for epoch in range(n_epochs):
        epoch_training_loss = train(
            model, train_dataloader, optimizer, criterion, epoch
        )
        training_loss.append(epoch_training_loss)
        epoch_validation_loss = validate(model, test_dataloader, criterion, epoch)
        validation_loss.append(epoch_validation_loss)

    print("Finished Training and Validating")
    return training_loss, validation_loss


# Adapted from https://github.com/nalbert9/Facial-Keypoint-Detection
def train(model, train_dataloader, optimizer, criterion, epoch):
    """Train the model

    Args:
        model (torch.nn.Module): The PyTorch model
        train_dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader for training data
        optimizer (torch.optim.Optimizer): The PyTorch Optimizer 
        criterion (torch.nn.MSELoss): The PyTorch Criterion
        epoch (int): The current epoch number

    Returns:
        float: average loss
    """    
    model.train()

    running_loss = 0.0

    for batch_i, data in enumerate(train_dataloader):
        image = data["image"]
        key_pts = data["landmarks"]

        key_pts = key_pts.view(key_pts.size(0), -1)

        # convert variables to floats for regression loss
        key_pts = (
            key_pts.type(torch.cuda.FloatTensor)
            if torch.cuda.is_available()
            else key_pts.type(torch.FloatTensor)
        )
        image = (
            image.type(torch.cuda.FloatTensor)
            if torch.cuda.is_available()
            else image.type(torch.FloatTensor)
        )

        # zero the parameter (weight) gradients
        optimizer.zero_grad()

        # forward pass to get outputs
        output_pts = model(image)

        # calculate the loss between predicted and target keypoints
        loss = criterion(output_pts, key_pts)

        # backward pass to calculate the weight gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # print loss statistics
        running_loss += loss.item()
        if batch_i % 10 == 9:  # print every 10 batches
            print(
                f"Epoch: {epoch + 1}, Batch: {batch_i + 1}, Avg. Training Loss: {running_loss / (batch_i + 1)}"
            )
    return running_loss / (batch_i + 1)


# Inspired from https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
def validate(model, test_dataloader, criterion, epoch):
    """Validate the trained model

    Args:
        model (torch.nn.Module): The PyTorch model
        test_dataloader (torch.utils.data.DataLoader): The PyTorch DataLoader for test data
        criterion (torch.nn.MSELoss): The PyTorch Criterion
        epoch (int): The current epoch number

    Returns:
        float: average loss
    """

    model.eval()

    running_loss = 0.0

    with torch.no_grad():
        for batch_i, data in enumerate(test_dataloader):
            image = data["image"]
            key_pts = data["landmarks"]

            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = (
                key_pts.type(torch.cuda.FloatTensor)
                if torch.cuda.is_available()
                else key_pts.type(torch.FloatTensor)
            )
            image = (
                image.type(torch.cuda.FloatTensor)
                if torch.cuda.is_available()
                else image.type(torch.FloatTensor)
            )

            outputs = model(image)
            loss = criterion(outputs, key_pts)
            running_loss += loss.item()

            print(
                f"Epoch: {epoch + 1}, Batch: {batch_i + 1}, Avg. Validation Loss: {running_loss / (batch_i + 1)}"
            )

    print("Finished Validating")
    return running_loss / (batch_i + 1)


def get_predicted_keypoints(
    image,
    transform,
    model,
    bbox=None,
    num_keypoints=58,
    post_transform=lambda predicted_key_pts, bbox: predicted_key_pts * 4,
):
    """Invokes the model to get keypoint predictions

    Args:
        image (ndarray): The input image
        transform (object): A callable PyTorch transform class
        model (torch.nn.Module): The PyTorch model
        bbox (list[float], optional): A list of coordinates of the bounding box of the face ([top, left, width, height]). Defaults to None.
        num_keypoints (int, optional): Number of keypoints expected. Defaults to 58.
        post_transform (function, optional): A callback function to transform the predicted keypoints before returning.

    Returns:
        ndarray: predicted keypoints
    """    

    sample = {"image": image, "landmarks": np.array([])}
    if bbox is not None:
        sample["bbox"] = bbox

    sample = transform(sample)

    sample_image = sample["image"]

    with torch.no_grad():
        key_points = model(sample_image.unsqueeze(0))
    key_points = key_points.view(key_points.size()[0], -1, num_keypoints)

    # transform key points back to match the 640x480 photo
    predicted_key_pts = key_points.data.cpu().numpy()[0]
    predicted_key_pts = post_transform(predicted_key_pts, bbox)

    return predicted_key_pts


def download_ibug_face_database():
    """
    Downloads the IBug Face Database - https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    WARNING! This is a large file @ ~2GB
    """
    if not os.path.exists("content/ibug_300W_large_face_landmark_dataset"):
        url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"

        r = requests.get(url, allow_redirects=True)
        with open("content/ibug_300W_large_face_landmark_dataset.tar.gz", "wb") as file:
            file.write(r.content)

        with tarfile.open(
            "content/ibug_300W_large_face_landmark_dataset.tar.gz"
        ) as file:
            file.extractall("content")
