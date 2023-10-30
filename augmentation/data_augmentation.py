import numpy as np
import random
from mayavi import mlab
import os
from volumentations import *
from volumentations.augmentations import transforms as ai

"package used: https://github.com/ZFTurbo/volumentations/blob/master/volumentations/augmentations/transforms.py"

class field_augmentation(object):
    def __init__(self,duplicate,crop,rotation,translation,rand_noise,inver_intensity,rand_flip,contrast):
        self.duplicate = duplicate
        self.crop=crop
        self.rotation=rotation
        self.translation=translation
        self.rand_noise=rand_noise
        self.inver_intensity=inver_intensity
        self.rand_flip=rand_flip
        self.contrast=contrast
        self.flag=False
        self.transformer=[]
    def __call__(self, grid):
        grid = self.self_augmentation(grid)
        augmented_data = self.vol_augmentation()(image=grid)["image"]
        return augmented_data

    def grid_duplicate(self,grid,p=0.6):
        tmp = random.uniform(0, 1)
        if tmp < p:
            self.flag = True

            # Randomly choose between 8-cell or 27-cell duplication with 2 : 1 ratio
            choice = random.choice([2,2,3])

            # Tile the grid
            grid = np.tile(grid, (choice, choice, choice))

            # Compute the mean of each (choice x choice x choice) block to downsample
            grid = grid.reshape(grid.shape[0] // choice, choice,
                                grid.shape[1] // choice, choice,
                                grid.shape[2] // choice, choice).mean(axis=(1, 3, 5))
        return grid

    def grid_translation(self,grid,p=1):
        tmp = random.uniform(0, 1)
        if tmp<p:
            dx = random.randint(0,grid.shape[0])
            dy = random.randint(0, grid.shape[1])
            dz = random.randint(0, grid.shape[2])
            grid = np.roll(grid,(dx,dy,dz),axis=(0,1,2))
        return grid


    def grid_inverse_intensity(self,grid,p=0.5):
        tmp=random.uniform(0,1)
        if tmp<p:
            grid=1-grid
        return grid

    def vol_augmentation(self):
        if self.rand_flip:
            tf = ai.Flip(p=0.5)
            self.transformer.append(tf)
        if self.rotation:
            tf=ai.Rotate(p=0.8,border_mode='wrap',x_limit=[-45,45],y_limit=[-45,45],z_limit=[-45,45])
            self.transformer.append(tf)
        if self.rand_noise:
            tmp = random.uniform(0,0.2)
            tf = ai.GaussianNoise(var_limit=(0,tmp),mean=0,p=1)
            self.transformer.append(tf)
        ##need to convert the dtype of input image,do not change the p unless the conversion is included
        if self.contrast:
            tf = ai.RandomBrightnessContrast(p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True)
            self.transformer.append(tf)
        if self.flag and self.crop:
            tf =ai.RandomResizedCrop(shape=(32,32,32),scale_limit=(0.125, 1),resize_type=0,p=0.5)
            self.transformer.append(tf)
        return Compose(self.transformer)

    def self_augmentation(self,grid):
        if self.translation:
            grid=self.grid_translation(grid,p=1)
        if self.duplicate:
            grid=self.grid_duplicate(grid,p=0.6)
        if self.inver_intensity:
            grid = self.grid_inverse_intensity(grid,p=0.5)
        return grid

if __name__=="__main__":
    def visualize_with_mlab(values, fig_num=1):
        mlab.figure(fig_num, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
        mlab.contour3d(values, contours=[0.5], transparent=False)
        mlab.show()

    sample_object = field_augmentation(duplicate=False,crop=False,rotation=False,
                                       translation=True,rand_noise=True, inver_intensity=False,
                                       rand_flip=True,contrast=True)
    test= np.loadtxt("SG.rf", skiprows=15)[:, 0].reshape(32, 32, 32)
    data = sample_object.self_augmentation(test)
    result = sample_object.vol_augmentation()(image=data)["image"]
    # Visualize the original data
    # visualize_with_mlab(test, fig_num=1)
#
#     # Visualize the augmented data

    visualize_with_mlab(result, fig_num=1)
    np.savetxt("./dg_no_rotation.rf",result.reshape(-1,1),fmt="%.6f")


