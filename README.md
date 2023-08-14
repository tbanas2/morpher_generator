# ComputationalPhotography-Course-Project

## Computational Photography Course Project- Spring2023

### Project Team

* Luke Banaszak
* Sakshi Maheshwari
* Rick Suggs
* Yitian Tang

### Table of Contents

#### [Installing Dependencies](#installing-dependencies)

#### [Modules](#modules)

#### [References](#references)

### Installing Dependencies

The project was created with Python 3.10 and has not been tested with any other version.

The Python dependencies can be installed by running the following command in the project directory.

```shell
pip install -r requirements.txt
```

Training PyTorch models can take a long time. If your computer is equipped with a GPU, that time
could be reduced significantly. However, you may need to install different dependencies. See
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for instructions on installing
GPU support on your computer.

For example, on Windows with Nvidia GPU, CUDA support can be enabled by first uninstalling
Pytorch, then reinstalling with the compiled CUDA binaries:

```shell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

### Modules

#### Final_notebook.ipynb

The notebook begins with a simple set of cells to use the product's main functionality to generate a morphing GIF from a directory of jpg images.

The notebook then provides end to end examples of training the PyTorch models, all the way through generating a morph sequence from automatically predicted keypoints from multiple images.

#### morpher.py

Class definition that generates a morph sequence from a directory of images.

#### NeuralNet/train_cnn_resnet50.ipynb

Notebook covers results from training the pretrained Resnet 50 PyTorch model on the IMM Image Database
Our trained model file for the IMM Image Database is larger than the Github file limit, but can be downloaded from [here](https://drive.google.com/file/d/18B-OyNvhSzki7BPGcgzI2pi1BBssM89b/view?usp=sharing)

#### NeuralNet/train_cnn_resnet50_ibug.ipynb

Notebook covers results from training the pretrained Resnet 50 PyTorch model on the IBug Image Database
Our trained model file for the IBug Image Database is larger than the Github file limit, but can be downloaded from [here](https://drive.google.com/file/d/1luxLEy6aK5yr1dDLNwNMVpOQown8Hsr9/view)

#### NeuralNet/cnn_utils.py

Module of utility functions used for training PyTorch models.

#### NeuralNet/face_classifier.py

Module of PyTorch Transforms and DataLoaders used for training PyTorch models.

#### NeuralNet/Models/resnet.py

Class definitions for pretrained Resnet 50 PyTorch models.

#### NeuralNet/Models/simpleModel.py

Class definitions for custom PyTorch models.

#### FaceMorphing/morphing.ipynb

Notebook covers manually defined correspondence points, creating morphing animations from multiple images, and experimenting with a population of images.

#### FaceMorphing/morphing_utils.py

Module of utility functions used for warping and morphing.

#### facial_keypoint_detection/1_load_and_visualize_data.ipynb

Notebook covers loading and visualizing data from the YouTube Faces Dataset

#### facial_keypoint_detection/2_define_the_network_architecture.ipynb

Notebook covers PyTorch CNN instantiation for YouTube Faces Dataset

#### facial_keypoint_detection/3_facial_keypoint_detection_complete_pipeline.ipynb

Notebook covers PyTorch CNN completed pipeline for YouTube Faces Dataset

#### facial_keypoint_detection/data_load.py

Module for YouTube Faces Dataset PyTorch DataLoader

#### facial_keypoint_detection/models.py

Module for YouTube Faces Dataset PyTorch Models

### References

#### Course Content from CS445

[Lesson 6.1 - Image Morphing](https://www.coursera.org/learn/cs-445/home/week/6)

#### Programming Assignments from other Computational Photography courses

[UC Berkeley: Facial Keypoint Detection with Neural Networks](https://inst.eecs.berkeley.edu/~cs194-26/fa22/hw/proj5/)

[UC Berkeley: Face Morphing](https://inst.eecs.berkeley.edu/~cs194-26/fa22/hw/proj3/)

#### Papers

[[1805.04140] Neural Best-Buddies: Sparse Cross-Domain Correspondence](https://arxiv.org/abs/1805.04140)

[[1512.03385] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

#### Open Source Datasets

[The IMM Face Database - An Annotated Dataset of 240 Face Images](https://web.archive.org/web/20200207215115/http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id%3D3160) ([download links](https://web.archive.org/web/20210305094647/http://www2.imm.dtu.dk/~aam/datasets/datasets.html))

[Resnet50 Pretrained PyTorch Model](https://github.com/Cadene/pretrained-models.pytorch#torchvision)

[i·bug - resources - Facial point annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

#### Open Source Code

[nalbert9/Facial-Keypoint-Detection: Computer vision](https://github.com/nalbert9/Facial-Keypoint-Detection)

[Neural Best-Buddies: Sparse Cross-Domain Correspondence](https://kfiraberman.github.io/neural_best_buddies/)

#### Tutorials & Blog Articles

[Neural Networks — PyTorch Tutorials 2.0.0+cu117 documentation](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

[Writing Custom Datasets, DataLoaders and Transforms — PyTorch Tutorials 2.0.0+cu117 documentation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight%3Ddataloader)

[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

[Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

[Facial KeyPoint Detection with Pytorch | by Antonio Linares | Analytics Vidhya | Medium](https://medium.com/analytics-vidhya/facial-keypoint-detection-with-pytorch-e9f94ab321a2)

[Face Landmarks Detection With PyTorch | by Abdur Rahman Kalim | Towards Data Science](https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4)

[Setting the learning rate of your neural network.](https://www.jeremyjordan.me/nn-learning-rate/)

[Drawing Loss Curves for Deep Neural Network Training in PyTorch | by Niruhan Viswarupan](https://niruhan.medium.com/drawing-loss-curves-for-deep-neural-network-training-in-pytorch-ac617b24c388)

[Multivariate data interpolation on a regular grid (RegularGridInterpolator) — SciPy v1.10.1 Manual](https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html)

[An Introductory Guide to Deep Learning and Neural Networks (Notes from deeplearning.ai Course #1)](https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/)

[Improving Neural Networks – Hyperparameter Tuning, Regularization, and More (deeplearning.ai Course #2)](https://www.analyticsvidhya.com/blog/2018/11/neural-networks-hyperparameter-tuning-regularization-deeplearning/)

[A Comprehensive Tutorial to learn Convolutional Neural Networks from Scratch (deeplearning.ai Course #4)](https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/?utm_source%3Dblog%26utm_medium%3Dbuilding-image-classification-models-cnn-pytorch)

[Getting Started with Facial Keypoint Detection using Deep Learning and PyTorch](https://debuggercafe.com/getting-started-with-facial-keypoint-detection-using-pytorch/)

[Advanced Facial Keypoint Detection with PyTorch](https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/)

[OpenCV: Cascade Classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

#### StackOverflow

[plot training and validation loss in pytorch - Stack Overflow](https://stackoverflow.com/questions/74754493/plot-training-and-validation-loss-in-pytorch)

#### Wikipedia

[Convolutional neural network - Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)

#### API References

[scipy.interpolate.RegularGridInterpolator — SciPy v1.10.1 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)

[scipy.spatial.Delaunay — SciPy v1.10.1 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)

[Module: draw — skimage v0.20.0 docs](https://scikit-image.org/docs/stable/api/skimage.draw.html%23skimage.draw.polygon2mask)

[numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)

[numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)

#### Image sources (President Obama and Clooney images were snipped from the doc)

[https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/proj4/cs194-26-abc/](https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/proj4/cs194-26-abc/)
