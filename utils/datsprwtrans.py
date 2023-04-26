import torch
from scipy.io import loadmat
import os
from PIL import Image
import cv2
import numpy as np
import math


def collate_fn(batch):
    """Dataloader data gatherer."""
    
    return tuple(zip(*batch))

class PRWTransfer(torch.utils.data.Dataset):
    """Creates a PRW dataset for a Object Detection model."""
    
    def __init__(self, img_root, ann_root, transforms=None):
        """Instantiates the dataset."""
        
        # set the root directories for data.
        self.root_img = img_root
        self.root_ann = ann_root
        self.transforms = transforms
        
        # load all image files, sorting them to
        # ensure that they are aligned.
        self.imgs = list(sorted(os.listdir(img_root)))
        self.anns = list(sorted(os.listdir(ann_root)))

    def __getitem__(self, idx):
        """Gets a sample from the dataset."""
        
        # load images and bounding boxes.
        img_path = os.path.join(self.root_img, self.imgs[idx])
        ann_path = os.path.join(self.root_ann, self.anns[idx])
        ann = loadmat(ann_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # get the height and width of the image
        image_width = img.shape[1]
        image_height = img.shape[0]
        img = cv2.resize(img, (416, 416))
        img /= 255.0
        
        if 'box_new' in ann.keys():
            ann = ann['box_new']
        elif 'anno_file' in ann.keys():
            ann = ann['anno_file']
        elif 'anno_previous' in ann.keys():
            ann = ann['anno_previous']
        else:
            raise ValueError("Invalid Annotation Error")

        # get bounding box coordinates.
        bbox = [a[1:] for a in ann]
        for i in bbox:
            xmin = i[0]
            xmax = xmin + i[2]
            ymin = i[1]
            ymax = ymin + i[3]

            xmin = (xmin/image_width)*416
            xmax = (xmax/image_width)*416
            ymin = (ymin/image_height)*416
            ymax = (ymax/image_height)*416
            coor = [xmin, ymin, xmax, ymax]
            for ix, c in enumerate(coor):
                coor[ix] = math.floor(c) if c > 1 else c
                coor[ix] = math.ceil(c) if c < 0 else c
            i[:] = coor
        bbox = np.array(bbox)
        # store boxes as a tensor.
        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        # there is only one class.
        labels = torch.ones((len(bbox),), dtype=torch.int64)
        
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = torch.FloatTensor(boxes)
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply transform if required.
        if self.transforms:
            sample = self.transforms(image = img,
                                     bboxes = target['boxes'],
                                     labels = labels)
            print(sample['bboxes'])
            for i, b in enumerate(sample['bboxes']):
                new = [0, 0, 0, 0]
                for a in range(4):
                    new[a] = math.floor(b[a]) if b[a] > 1 else b[a]
                    new[a] = math.ceil(b[a]) if b[a] < 0 else b[a]
                sample['bboxes'][i] = tuple(new)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        """Gets the size of the dataset (number of samples)."""
        
        return len(self.imgs)
    