import FaceMorphing.morphing_utils as mu
import NeuralNet.cnn_utils as cu
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
import NeuralNet.face_classifier as fc
import os

# All code written by Luke Banaszak or Rick Suggs unless otherwise noted

composed = transforms.Compose(
    [fc.Crop(), fc.Rescale((224, 224)), fc.NormalizeZeroToOne(), fc.ToColor(), fc.ToTensor()]
)

class Morpher():

    def __init__(self,model,transformers = composed):
        '''
        Provides a simple programming interface to morph face images. 
        
        Args:
        model: the neural net model that will ID landmarks
        transformers: pytorch transformations needed to prepare a raw image for processing in the model

        Returns:
        the morpher interface
        '''
        self.model = model
        self.transformers = transformers
        self.model.eval()

    def make_gif(self, dir, filename, frames=30):
        '''
        Produce a gif morphing sequence built from the images within a specified directory.
        '''
        resized_images, keypoints = self.process_images(dir)
        print("generating morph gif animation")
        mu.generate_morph_gif_animation(resized_images, keypoints, filename=filename, num_frames_per_morph=frames)
    
    def make_mpeg(self, dir, filename, frames=30):
        '''
        Produce a mpeg morphing sequence built from the images within a specified directory.
        '''
        resized_images, keypoints = self.process_images(dir)
        print("generating morph mpeg animation")
        mu.generate_morph_mpeg_animation(resized_images, keypoints, filename, num_frames_per_morph=frames)
    
    def process_images(self, dir):
        corners = np.asarray([(0, 0), (0, 480 - 1), (640 - 1, 0), (640 - 1, 480 - 1)])
        keypoints=[]
        resized_images=[]
        images = self.load_images(dir)
        print(f"{len(images)} images found")
        for i, image in enumerate(images):
            print(f"predicting keypoints for image: {i}")
            
            # ensure image is 3 channels
            if (len(image.shape) == 2):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            prediction = self.predict(image)
            if prediction is None:
                continue
            
            resized_images.append(mu.resize_image(image))
            print(f"resized image from {image.shape} to {resized_images[-1].shape}")
            points = np.asarray(prediction.T)
            points = np.concatenate((corners,points))
            keypoints.append(points)
        return resized_images, keypoints
    
    def post_transform(self, keypoints, bbox):
        # un-rescale from (224, 224)
        keypoints_t = keypoints.T
        keypoints_t = keypoints_t * np.array([bbox[2] / 224, bbox[3] / 224])
        # un-crop
        keypoints_t = keypoints_t + np.array([[bbox[1], bbox[0]]])
        return keypoints_t.T
    
    def predict(self,image):
        image = mu.resize_image(image)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # TODO: equalizing histogram causes some faces to not be detected
        # gray_image = cv2.equalizeHist(gray_image)

        faces = face_cascade.detectMultiScale(image_gray)

        if len(faces) == 0:
            print('no faces detected - skipping image')
            return None

        left, top, width, height = faces[0]
        bbox = [top, left, width, height]
        
        predicted_key_pts = cu.get_predicted_keypoints(
            image_rgb.astype(np.float32),
            self.transformers,
            self.model,
            bbox=bbox,
            num_keypoints=68,
            post_transform=self.post_transform,
        )
        return predicted_key_pts
    
    def predict_and_show(self,image):
        points = self.predict(image)
        plt.imshow(mu.resize_image(image))
        plt.scatter(*points, s=5, marker=".")

    def load_images(self,dir):
        '''
        Import jpgs from a dir into a list.
        '''
        images = []
        for file in os.listdir(dir):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                print(f"reading image: {filename}") 
                image = plt.imread(os.path.join(dir, filename)).astype(np.uint8)
                images.append(image)
        return images
            

        
