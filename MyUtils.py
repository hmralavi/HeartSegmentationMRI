"""
 Created By Hamid Alavi on 7/3/2019
"""
import numpy as np
import keras
import nibabel as nib
import os.path as path
from scipy import interpolate
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf


def split_train_test_id(id_all_data, testing_share):
    _id = id_all_data.copy()
    r = np.random.RandomState(seed=1000)
    r.shuffle(_id)
    id_training_data = np.sort(_id[np.floor(len(id_all_data) * testing_share).astype(np.uint16):])
    id_testing_data = np.sort(_id[:np.floor(len(id_all_data) * testing_share).astype(np.uint16)])
    return id_training_data, id_testing_data


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path, patient_ids, batch_size=32, new_voxel_size=1.25, crop_size=(256, 256),
                 image_masks_index=(0, 1, 2, 3), shuffle=True):
        self.data_path = data_path
        self.patient_ids = patient_ids.copy()
        self.batch_size = batch_size
        self.new_voxel_size = new_voxel_size
        self.crop_size = crop_size
        self.image_masks_index = image_masks_index
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.patient_ids)

    def __len__(self):
        return int(np.floor(len(self.patient_ids) / self.batch_size))

    def _normalize_image(self, image):
        image = (image - image.mean()) / image.std()
        return image

    def _layers_single2multi(self, mask):
        new_mask = np.zeros((*mask.shape[0:2], len(self.image_masks_index)), dtype=np.int8)
        for i, layer_index in enumerate(self.image_masks_index):
            _layer = mask == layer_index
            new_mask[:, :, i] = _layer
        return new_mask

    def _layers_multi2single(self, mask):
        new_mask_shape = list(mask.shape)
        new_mask_shape[-1] = 1
        new_mask = np.zeros(new_mask_shape, dtype=np.int8)
        for j in range(mask.shape[0]):
            for i in range(mask.shape[-1]):
                new_mask[j, mask[j, :, :, i] == 1, 0] = self.image_masks_index[i]
        return new_mask

    def __getitem__(self, batch_index):
        if batch_index == self.__len__() - 1:
            _patient_ids = self.patient_ids[batch_index * self.batch_size:]
        else:
            _patient_ids = self.patient_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        images = []
        masks = []
        for p_id in _patient_ids:
            pfolder = path.join(self.data_path, 'patient' + str(p_id).zfill(3))
            with open(path.join(pfolder, 'info.cfg')) as info:
                lines = info.readlines()
                ed = int(lines[0].split()[1])
                es = int(lines[1].split()[1])
            for frame in [ed, es]:
                inputfilename = path.join(pfolder,
                                          'patient' + str(p_id).zfill(3) + '_frame' + str(frame).zfill(2) + '.nii.gz')
                inputfilename_mask = path.join(pfolder, 'patient' + str(p_id).zfill(3) + '_frame' + str(frame).zfill(
                    2) + '_gt.nii.gz')
                data3D = nib.load(inputfilename)
                data3D_mask = nib.load(inputfilename_mask)
                voxel_size = data3D.header.get_zooms()[0]
                FOV = (np.array(self.crop_size)) * self.new_voxel_size / 2
                img_size = np.array(data3D.shape[0:2]) + 2
                img_center = voxel_size * (img_size - 1) / 2;
                xgrid, ygrid = np.mgrid[0:voxel_size * (img_size[0]):voxel_size,
                               0:voxel_size * (img_size[1]):voxel_size]
                xgridnew, ygridnew = np.mgrid[img_center[0] - FOV[0]:img_center[0] + FOV[0]:self.new_voxel_size,
                                     img_center[1] - FOV[1]:img_center[1] + FOV[1]:self.new_voxel_size]
                for s in range(data3D.shape[2]):
                    img_raw = np.pad(np.squeeze(data3D.slicer[:, :, s:s + 1].get_fdata()), ((1, 1), (1, 1)),
                                     constant_values=0)
                    mask_raw = np.pad(np.squeeze(data3D_mask.slicer[:, :, s:s + 1].get_fdata()), ((1, 1), (1, 1)),
                                      constant_values=0)
                    img_cropped = interpolate.griddata((xgrid.flat[:], ygrid.flat[:]), img_raw.flat[:],
                                                       (xgridnew, ygridnew), method='nearest', fill_value=0)
                    mask_cropped = interpolate.griddata((xgrid.flat[:], ygrid.flat[:]), mask_raw.flat[:],
                                                        (xgridnew, ygridnew), method='nearest', fill_value=0)
                    images.append(self._normalize_image(np.expand_dims(img_cropped, axis=-1)))
                    masks.append(self._layers_single2multi(mask_cropped))
        images = np.array(images)
        masks = np.array(masks)
        return images, masks


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels

def dice_loss_multilabel(y_true, y_pred, numLabels=4):
    return 1 - dice_coef_multilabel(y_true, y_pred, numLabels)