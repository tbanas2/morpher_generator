import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from skimage.draw import polygon2mask
from scipy.interpolate import RegularGridInterpolator
from os import path
import requests
import tarfile
import imageio
import ffmpeg
import cv2
from skimage import transform

# Morphing Utility Functions
# All code written by Rick Suggs, Sakshi Maheshwari, or Luke Banaszak unless otherwise noted.


def choose_points(left_image, left_points, right_image, right_points, max_points):
    """
    Interactively choose correspondence points on two images

    Args:
        left_image (ndarray): left image in plot
        left_points (list): an empty list to store left image correspondence points
        right_image (ndarray): right image in plot
        right_points (list): an empty list to store right image correspondence points
        max_points (int): number of points to choose for each image
    """
    fig, axes = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))

    left, right = axes[0]
    left.imshow(left_image)
    right.imshow(right_image)

    def onclick(event):
        event_axes = event.inaxes
        points = left_points if event_axes == left else right_points

        ## only get specified points from the user
        if len(points) >= max_points:
            fig.canvas.mpl_disconnect(cid)
        else:
            event_axes.plot(event.xdata, event.ydata, "r+")
            points.append((event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # display plot
    plt.show()


def getavg(a, b):
    return a + 0.5 * (b - a)


def compute_affine(original_tri_pts, mean_tri_pts, use_lstsq=False):
    """
    Computes an affine transformation between two sets of corresponding points

    Args:
        original_tri_pts (ndarray): Original triangle points
        mean_tri_pts (ndarray): Mean triangle points
        use_lstsq (bool, optional): Use least squares solver instead of inverse matrix solution. Defaults to False.

    Returns:
        ndarray: A matrix representing an affine linear transformation between the two sets of points
    """
    A = np.zeros((6, 6))
    b = np.zeros(6)
    e = 0
    for original_tri_pt, mean_tri_pt in zip(original_tri_pts, mean_tri_pts):
        A[e, :3] = np.append(original_tri_pt, [1])
        b[e] = mean_tri_pt[0]
        e += 1

        A[e, 3:] = np.append(original_tri_pt, [1])
        b[e] = mean_tri_pt[1]
        e += 1

    x = np.linalg.lstsq(A, b)[0] if use_lstsq else np.linalg.solve(A, b)
    transform = np.append(x, [0, 0, 1]).reshape((3, 3))
    inv_transform = np.linalg.inv(transform)
    return inv_transform


def warp_image(original_image, original_points, mean_points, use_lstsq=False):
    """
    Warps an image using the two sets of correspondence points

    Args:
        original_image (ndarray): The image to warp
        original_points (ndarray): Correspondence points of the original image
        mean_points (ndarray): Mean points
        use_lstsq (bool, optional): Use least squares solver instead of inverse matrix solution in compute_affine. Defaults to False.

    Returns:
        ndarray: The resulting warped image
    """

    triangles = Delaunay(mean_points)
    mean_tri_indices = triangles.simplices

    warped_image = np.zeros_like(original_image)

    height = original_image.shape[0]
    width = original_image.shape[1]

    xs = np.arange(height)
    ys = np.arange(width)
    interp = RegularGridInterpolator(
        (xs, ys), original_image, bounds_error=False, fill_value=None
    )

    for mean_triangle in mean_tri_indices:
        original_tri_pts = original_points[mean_triangle]
        mean_tri_pts = mean_points[mean_triangle]

        affine_tx = compute_affine(original_tri_pts, mean_tri_pts, use_lstsq=use_lstsq)

        mask = polygon2mask((height, width), mean_tri_pts)

        X, Y = mask.nonzero()
        original_x, original_y, _ = np.dot(
            affine_tx, np.vstack([X, Y, np.ones(X.shape[0])])
        )
        orig_points = np.column_stack((original_x, original_y))
        warped_image[X, Y, :] = interp(orig_points)

    return warped_image


def morph(im1, im2, im1_pts, im2_pts, warp_frac, dissolve_frac):
    """
    Warps two images to a set of mean points between the correspondence points of the two images
    Cross dissolves the images.

    Args:
        im1 (ndarray): Image A
        im2 (ndarray): Image B
        im1_pts (ndarray): Points for Image A
        im2_pts (ndarray): Points for Image B
        warp_frac (float): Warp fraction
        dissolve_frac (float): Dissolve fraction

    Returns:
        ndarray: The morphed image
    """
    # Compute intermediate shape configuration
    mean_pts = (1 - warp_frac) * im1_pts + warp_frac * im2_pts

    # Warp images to intermediate shape configuration
    warp_im1 = warp_image(im1, np.flip(im1_pts), np.flip(mean_pts))
    warp_im2 = warp_image(im2, np.flip(im2_pts), np.flip(mean_pts))

    # Cross-dissolve warped images
    morphed_im = (1 - dissolve_frac) * warp_im1 + dissolve_frac * warp_im2

    return morphed_im


def get_pts_from_asf(path2file, shape, is_add_corners=True):
    """
    Extracts points from an ASF file

    Args:
        path2file (string): Path to the file
        shape (tuple): Image shape
        is_add_corners (bool, optional): Adds the image corner points. Defaults to True.

    Returns:
        ndarray: a data structure of the keypoints parsed from the asf file
    """
    with open(path2file, "r") as f:
        data = f.readlines()[16:74]

    pts_rel = np.array([data[i].split("\t")[2:4] for i in range(len(data))]).astype(
        float
    )

    pts_real = np.multiply(np.array([shape[1], shape[0]]), pts_rel)
    if is_add_corners:
        corners = [
            [0, 0],
            [shape[1] - 1, 0],
            [0, shape[0] - 1],
            [shape[1] - 1, shape[0] - 1],
        ]
        pts_real = np.vstack([pts_real, np.array(corners)])
    return pts_real


def mean_images(images, image_pts):
    """
    Returns a list of images warped to the mean pts

    Args:
        images (list[ndarray]): a list of images
        image_pts (list[ndarray]): a list of lists of correspondence points

    Returns:
        list<ndarray>: a list of images warped to the mean pts
    """
    mean_pts = np.mean(image_pts, axis=0)
    return [
        warp_image(images[i], np.flip(image_pts[i]), np.flip(mean_pts))
        for i in range(len(images))
    ]


def blend_images(images):
    """
    Generates a image blending all images in the images param

    Args:
        images (list[ndarray]): a list of images

    Returns:
        ndarray: an image blending the input images
    """
    dissolve_frac = 1 / len(images)
    return np.sum(np.array([image * dissolve_frac for image in images]), axis=0)


def mean_face(images, image_pts):
    """
    Creates the mean image from a list of images and their correspondences points

    Args:
        images (list[ndarray]): a list of images
        image_pts (list[ndarray]): a list of lists of correspondence points

    Returns:
        ndarray: The mean image
    """
    warped_images = mean_images(images, image_pts)
    return blend_images(warped_images)


def download_imm_face_database():
    """
    Downloads the IMM Face Database - An Annotated Dataset of 240 Face Images
    """
    dir_db = "data/"
    dir_imm_face_db = f"{dir_db}imm_face_db/"
    filename = "imm_face_db.tar.gz"
    if not path.exists(f"{dir_db}{filename}"):
        url = f"https://web.archive.org/web/20210305094647/http://www2.imm.dtu.dk/~aam/datasets/{filename}"

        r = requests.get(url, allow_redirects=True)
        with open(f"{dir_db}{filename}", "wb") as file:
            file.write(r.content)

        with tarfile.open(f"{dir_db}{filename}") as file:
            file.extractall(dir_imm_face_db)


def generate_morph_gif_animation(images, keypoints, filename, num_frames_per_morph=30):
    """
    Creates a morph sequence from the list of images, based on the list of corresponding keypoints.
    Exports a gif animation of the resulting morph sequence.

    Args:
        images (list[ndarray]): List of images
        keypoints (list[ndarray]): List of list of keypoints corresponding to the images
        filename (string): Output path & filename
        num_frames_per_morph (int, optional): The number of steps in the morph between each image. Defaults to 30.
    """    
    with imageio.get_writer(filename, mode="I", duration=0.1, loop=0) as writer:
        for i in range(len(images) - 1):
            image_1 = images[i]
            image_2 = images[i + 1]
            keypoints_1 = keypoints[i]
            keypoints_2 = keypoints[i + 1]
            for j in range(num_frames_per_morph):
                warp_frac = j / (num_frames_per_morph - 1)
                dissolve_frac = j / (num_frames_per_morph - 1)
                morphed_im = morph(
                    image_1, image_2, keypoints_1, keypoints_2, warp_frac, dissolve_frac
                )
                writer.append_data(morphed_im.astype(np.uint8))


def generate_morph_mpeg_animation(images, keypoints, filename, num_frames_per_morph=30):
    """
    Creates a morph sequence from the list of images, based on the list of corresponding keypoints.
    Exports a mpeg animation of the resulting morph sequence.

    Args:
        images (list[ndarray]): List of images
        keypoints (list[ndarray]): List of list of keypoints corresponding to the images
        filename (string): Output path & filename
        num_frames_per_morph (int, optional): The number of steps in the morph between each image. Defaults to 30.
    """     
    morphed_images = []
    for i in range(len(images) - 1):
        image_1 = images[i]
        image_2 = images[i + 1]
        keypoints_1 = keypoints[i]
        keypoints_2 = keypoints[i + 1]
        for j in range(num_frames_per_morph):
            warp_frac = j / (num_frames_per_morph - 1)
            dissolve_frac = j / (num_frames_per_morph - 1)
            morphed_im = morph(
                image_1, image_2, keypoints_1, keypoints_2, warp_frac, dissolve_frac
            )
            morphed_images.append(morphed_im.astype(np.uint8))

    vidwrite_from_numpy(filename, morphed_images)


# function copied from from CS445 project 5
def vidwrite_from_numpy(filename, images, framerate=30, vcodec="libx264"):
    """
    Writes a file from a numpy array of size nimages x height x width x RGB
    # source: https://github.com/kkroening/ffmpeg-python/issues/246
    """
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height)
        )
        .output(filename, pix_fmt="yuv420p", vcodec=vcodec, r=framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def resize_image(img, size=(640, 480)):
    """
    Resizes an image

    Args:
        img (ndarray): The image to resize
        size (tuple, optional): The dimensions to resize to (width, height). Defaults to (640, 480).

    Returns:
        ndarray: The resized image
    """    
    img = crop_to_aspect(img)
    height, width = img.shape[:2]
    max_height = size[1]
    max_width = size[0]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(
            img,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
    elif max_height > height or max_width > width:
        # expand image if less than desired size
        img = cv2.resize(img, size)
    return img


def crop_to_aspect(img):
    """
    Crops an images to the ideal aspect ratio (4:3)

    Args:
        img (ndarray): The image to crop

    Returns:
        ndarray: The cropped image
    """    
    img = np.asarray(img)
    height, width = img.shape[:2]
    aspect = width / float(height)
    ideal_width = 640
    ideal_height = 480
    ideal_aspect = ideal_width / float(ideal_height)
    if aspect > ideal_aspect:
        # Then crop the left and right edges:
        new_width = int(ideal_aspect * height)
        offset = int((width - new_width) / 2)
        return img[:, offset : (width - offset)]
    else:
        # ... crop the top and bottom:
        new_height = int(width / ideal_aspect)
        offset = int((height - new_height) / 2)
        return img[offset : (height - offset), :]
