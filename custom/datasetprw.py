from torch.utils.data import Dataset
import torch
import os
import cv2 as cv
from scipy.io import loadmat
import numpy as np
from operator import itemgetter


def generate_patch(scale=12):
    """Creates a heatmap using Gaussian Distribution."""

    # constants.
    sigma = 1

    size = 6 * sigma + 1
    x_mesh, y_mesh = torch.meshgrid(torch.arange(0, 6*sigma+1, 1), torch.arange(0, 6*sigma+1, 1), indexing='xy')

    # the center of the gaussian patch should be 1
    center_x = size // 2
    center_y = size // 2

    # generate this 7x7 gaussian patch
    xmesh = torch.square(torch.sub(x_mesh, center_x))
    ymesh = torch.square(torch.sub(y_mesh, center_y))
    denom = (sigma**2) * 2
    gaussian_patch = torch.mul(torch.exp(torch.div(torch.neg(torch.add(xmesh, ymesh)), denom)), scale)

    return gaussian_patch


def make_heatmap_plural(width, height, keypoints, gau_patch):
    """Places a Gaussian Patch in the heatmap."""

    # constants.
    heatmap = np.zeros((height, width))
    sigma = 1
    visibility = 2
    gau_patch = torch.Tensor.numpy(gau_patch)

    # generates the limits of the patch for each keypoint.
    coordinates = []
    for keypoint in keypoints:
        center_x, center_y = keypoint
        
        # get the coordinates.
        xmin = center_x - 3 * sigma
        ymin = center_y - 3 * sigma
        xmax = center_x + 3 * sigma
        ymax = center_y + 3 * sigma
        coordinates.append((xmin, ymin, xmax, ymax))

    # for each keypoint draw the patch.
    for coordinate in coordinates:
        # unpack the coordinates.
        xmin, ymin, xmax, ymax = coordinate

        # if outside the image don't include the patch.
        if xmin >= width or ymin >= height or xmax < 0 or ymax < 0 or visibility == 0:
            pass

        # determine boundaries for patch if outside the image.
        patch_xmin = max(0, -xmin)
        patch_ymin = max(0, -ymin)
        patch_xmax = min(xmax, width) - xmin
        patch_ymax = min(ymax, height) - ymin

        # we need to determine where to put this patch in the whole heatmap
        heatmap_xmin = int(max(0, xmin))
        heatmap_ymin = int(max(0, ymin))

        # add the patches to the image.
        for j in range(int(patch_ymin), int(patch_ymax)):
            for i in range(int(patch_xmin), int(patch_xmax)):
                
                # get the value within the patch.
                gau_pixel = gau_patch[j, i]
                pixel = heatmap[j+heatmap_ymin, i+heatmap_xmin]
                
                # if the pixel already has a value assigned to it take the max.
                if pixel > 0:
                    heatmap[j+heatmap_ymin, i+heatmap_xmin] = max(pixel, gau_pixel)
                else:
                    heatmap[j+heatmap_ymin, i+heatmap_xmin] = gau_pixel

    # return the final heatmap as a tensor.
    return torch.FloatTensor(heatmap)


class PrwHeatMaps(Dataset):
    """Creates a custom dataset with the PRW data."""

    def __init__(self, img_path, ann_path, indexes, transform=None, resize_shape=(64, 64)):
        """Instantiates the dataset."""

        # setup transforms
        self.transform = transform
        self.resize_shape = resize_shape

        # set up img and annotations paths.
        self.img_path = img_path
        self.ann_path = ann_path

        # get all image and annotations.
        self.img_names = sorted(list(os.listdir(img_path)))
        self.ann_names = sorted(list(os.listdir(ann_path)))
        
        # get the given index range. (This is for train/testing)
        self.img_names = itemgetter(*indexes)(self.img_names)
        self.ann_names = itemgetter(*indexes)(self.ann_names)
        
        # create a patch.
        self.patch = generate_patch(1)

        # check that the annotations and images match.
        if len(self.img_names) != len(self.ann_names):
            raise ValueError("Images and annotations don't align")
        for img, ann in zip(self.img_names, self.ann_names):
            name, _ = os.path.splitext(img)
            if name not in ann:
                raise ValueError("Image and annotation names don't align")

    def _get_ann(self, name):
        """Loads in the annotation and gets the bounding boxes."""

        # load in the annotation.
        name = os.path.join(self.ann_path, name)
        ann = loadmat(name)
        if 'box_new' in ann.keys():
            ann = ann['box_new']
        elif 'anno_file' in ann.keys():
            ann = ann['anno_file']
        else:
            raise ValueError("Invalid Annotation Error")
        # remove the first value of each set of bounding boxes.
        ann = [a[1:] for a in ann]

        centers = list()
        # calculate the center of the bounding box.
        for a in ann:
            # unpack the annotation.
            x, y, w, h = a
            centers.append((x+(w/2), y+(h/2)))

        return centers

    def _get_img(self, name):
        """Loads in an image."""

        name = os.path.join(self.img_path, name)
        photo = cv.cvtColor(cv.imread(name), cv.COLOR_BGR2RGB)
        height, width = photo.shape[0], photo.shape[1]

        return photo, width, height

    def __len__(self):
        """Returns the total number of samples"""

        return len(self.img_names)

    def __getitem__(self, index):
        """Returns one sample of the data."""

        # get the image and the corresponding annotation.
        img, width, height = self._get_img(self.img_names[index])
        ann = self._get_ann(self.ann_names[index])
        count = len(ann)

        # convert the annotation to tensor.
        # ann = torch.FloatTensor(ann)

        # apply transform if available.
        if self.transform:
            # get the center of mass.
            rx, ry = self.resize_shape[0]/width, self.resize_shape[1]/height
            ann = [(a[0]*rx, a[1]*ry) for a in ann]

            # create the patch and the heatmap.
            heat = make_heatmap_plural(self.resize_shape[0],
                                       self.resize_shape[1], 
                                       ann, 
                                       self.patch)

            return self.transform(img), heat, count

        return img, ann, count