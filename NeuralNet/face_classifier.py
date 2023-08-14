from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import xml.etree.ElementTree as ET

# CNN Utility Functions
# Code written by Luke Banaszak or Rick Suggs unless otherwise noted


def build_frame(directory, imagedir, header, skiprows) -> pd.DataFrame:
    """
    Consolidates all the ASF file data in a directory into a single dataframe
    Arguments:
        frame (pd.DataFrame)): Dataframe containing the landmark data
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    Returns:
        pandas dataframe containing the data
    """
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".asf"):
            person = filename[0:2]
            view = filename[3]
            gender = filename[4]
            data = pd.read_csv(
                os.path.join(imagedir, filename),
                names=header,
                skiprows=skiprows,
                delim_whitespace=True,
                index_col=False,
            )
            data["img"] = i
            data["person"] = person
            data["view"] = view
            data["gender"] = gender
            if "frame" in locals():
                frame = pd.concat([frame, data])
            else:
                frame = data.copy()
            i += 1
        else:
            continue
    frame.set_index(["img"], inplace=True)
    return frame


# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, frame, root_dir, transform=None):
        """
        Arguments:
            frame (pd.DataFrame)): Dataframe containing the landmark data
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame.index.unique())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current = self.landmarks_frame.loc[[idx], ["person", "view", "gender"]]
        file_data = current.iloc[0]
        person = int(file_data["person"])
        gender = file_data["gender"]
        view = int(file_data["view"])
        img_name = os.path.join(
            self.root_dir, "{:02d}-{:d}{}.jpg".format(person, view, gender)
        )
        image = np.asarray(io.imread(img_name), dtype="float32")
        landmarks = self.landmarks_frame.loc[idx]
        H, W, C = image.shape
        landmarks = np.asarray(
            [(landmarks["xrel"] * W).round(), (landmarks["yrel"] * H).round()],
            dtype="float",
        )
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Parsing code inspired from https://inst.eecs.berkeley.edu/~cs194-26/fa22/hw/proj5/
class IBugFaceLandmarksDataset(Dataset):
    def __init__(self, transform=None):
        tree = ET.parse(
            "content/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"
        )
        root = tree.getroot()
        root_dir = "content/ibug_300W_large_face_landmark_dataset/"

        self.bboxes = []
        self.landmarks = []
        self.img_filenames = []

        self.transform = transform

        for filename in root[2]:
            img_filename = os.path.join(root_dir, filename.attrib["file"])
            box = filename[0].attrib

            # x, y for the top left corner of the box, w, h for box width and height
            bbox = [
                int(box["top"]),
                int(box["left"]),
                int(box["width"]),
                int(box["height"]),
            ]

            # TODO: rejecting bounding boxes with negative numbers for now, but how to deal with this?
            negative_count = 0
            for i in bbox:
                if i < 0:
                    negative_count += 1
            if negative_count > 0:
                continue

            self.img_filenames.append(img_filename)
            self.bboxes.append(bbox)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib["x"])
                y_coordinate = int(filename[0][num].attrib["y"])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(np.array(landmark).T)

        self.landmarks = np.array(self.landmarks).astype(np.float32)

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.img_filenames[index]).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.landmarks[index]

        sample = {"image": image, "landmarks": landmarks, "bbox": self.bboxes[index]}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * np.expand_dims(np.asarray([new_w / w, new_h / h]), 1)

        return {"image": img, "landmarks": landmarks}


class ToGray(object):
    """Image to grayscale and normalize to -0.5 to 0.5"""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image = color.rgb2gray(image)
        old_range = image.max() - image.min()
        new_range = 0.5 - -0.5
        new_value = lambda x: (((x - image.min()) * new_range) / old_range) + -0.5
        image = new_value(image)
        return {"image": image, "landmarks": landmarks}


class Normalize(object):
    """Normalize image to -0.5 to 0.5"""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        old_range = image.max() - image.min()
        new_range = 0.5 - -0.5
        new_value = lambda x: (((x - image.min()) * new_range) / old_range) + -0.5
        image = new_value(image)
        return {"image": image, "landmarks": landmarks}


class NormalizeZeroToOne(object):
    """Normalize image to 0 - 1 range"""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        image = image / 255
        return {"image": image, "landmarks": landmarks}


class ToColor:
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        return {"image": image, "landmarks": landmarks}


class Crop:
    def __call__(self, sample):
        image, landmarks, bbox = sample["image"], sample["landmarks"], sample["bbox"]

        top, left, width, height = bbox

        cropped_image = image[top : top + height, left : left + width]

        if np.any(landmarks):
            landmarks = landmarks.T
            landmarks = landmarks - np.array([[left, top]])
            landmarks = landmarks.T

        return {"image": cropped_image, "landmarks": landmarks, "bbox": bbox}

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        if torch.cuda.is_available():
            return {
                "image": torch.from_numpy(image).cuda(),
                "landmarks": torch.from_numpy(landmarks).cuda(),
            }
        else:
            return {
                "image": torch.from_numpy(image),
                "landmarks": torch.from_numpy(landmarks),
            }
