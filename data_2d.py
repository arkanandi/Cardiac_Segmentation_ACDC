import os
import random
import monai.transforms as mt
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def normalize(data):
    data = (data - data.mean()) / data.std()
    return data


class ACDC_2D(Dataset):
    def __init__(self, source, ind, Transform=None):
        # basic transforms
        self.loader = mt.LoadImaged(keys=["image", "mask"])
        self.add_channel = mt.AddChanneld(keys=["image", "mask"])
        self.spatial_pad = mt.SpatialPadD(keys=["image", "mask"], spatial_size=tar_shape, mode="edge")
        self.spacing = mt.Spacingd(keys=["image", "mask"], pixdim=(1.25, 1.25, -1.0), mode=("nearest", "nearest"))
        # index
        self.ind = ind
        # transform
        if Transform is not None:
            self.transform = Transform
        else:
            self.transform = mt.Compose([
                mt.SpatialPadD(keys=["image", "mask"], spatial_size=tar_shape, mode="edge"),
                mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
            ])

        # take the images
        source = Path(source)
        dirs = os.listdir(str(source))  # stores patient name
        all_data_ed = []
        all_data_ed_mask = []
        all_data_es = []
        all_data_es_mask = []
        for filenames in dirs:
            patient_path = Path(str(source), filenames)  # individual patient path
            patient_info = str(patient_path) + "/Info.cfg"  # patient information
            file = open(patient_info, 'r').readlines()
            ED_frame = int(file[0].split(":")[1])
            ES_frame = int(file[1].split(":")[1])
            ED = Path(str(patient_path), filenames + "_frame{:02d}.nii.gz".format(ED_frame))
            ES = Path(str(patient_path), filenames + "_frame{:02d}.nii.gz".format(ES_frame))
            ED_gt = Path(str(patient_path), filenames + "_frame{:02d}_gt.nii.gz".format(ED_frame))
            ES_gt = Path(str(patient_path), filenames + "_frame{:02d}_gt.nii.gz".format(ES_frame))
            all_data_ed.append(ED)
            all_data_ed_mask.append(ED_gt)
            all_data_es.append(ES)
            all_data_es_mask.append(ES_gt)

        if self.ind is not None:
            all_data_ed = [all_data_ed[i] for i in self.ind]
            all_data_ed_mask = [all_data_ed_mask[i] for i in self.ind]
            all_data_es = [all_data_es[i] for i in self.ind]
            all_data_es_mask = [all_data_es_mask[i] for i in self.ind]

        self.data = [all_data_ed, all_data_ed_mask, all_data_es, all_data_es_mask]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        ED_img, ED_mask, ES_img, ES_mask = self.data[0][idx], self.data[1][idx], self.data[2][idx], self.data[3][idx]
        # data dict
        ED_data_dict = {"image": ED_img,
                        "mask": ED_mask}
        ES_data_dict = {"image": ES_img,
                        "mask": ES_mask}
        # instead of returning both ED and ES, I have to return just a random choice between ED and ES(image and mask)
        datalist = [ED_data_dict, ES_data_dict]
        data_return = np.random.choice(datalist)
        data_return = self.loader(data_return)
        data_return = self.add_channel(data_return)
        data_return = self.spacing(data_return)
        data_return["image"] = normalize(data_return["image"])
        num_slice = data_return["image"].shape[3]
        random_slice = random.randint(0, num_slice - 1)
        data_return["image"] = data_return["image"][:, :, :, random_slice]
        data_return["image"] = normalize(data_return["image"])
        data_return["mask"] = data_return["mask"][:, :, :, random_slice]
        data_return = self.transform(data_return)
        return data_return


# target/crop shape for the images and masks when training
tar_shape = [352, 352]
crop_shape = [224, 224]


def train_loader_ACDC(train_index, data_path=r"D:\Master_Thesis\Lightning\training", transform=None):
    train_loader = ACDC_2D(source=data_path, Transform=transform, ind=train_index)
    return train_loader


def val_loader_ACDC(val_index, data_path=r"D:\Master_Thesis\Lightning\training", transform=None):
    val_loader = ACDC_2D(source=data_path, Transform=transform, ind=val_index)
    return val_loader


def test_loader_ACDC(test_index, data_path=r"D:\Master_Thesis\Lightning\testing", transform=None):
    test_loader = ACDC_2D(source=data_path, Transform=transform, ind=test_index)
    return test_loader


""" To test if the dataloader works """

train_compose = mt.Compose(
    [mt.SpatialPadD(keys=["image", "mask"], spatial_size=tar_shape, mode="edge"),
     mt.RandSpatialCropD(keys=["image", "mask"], roi_size=crop_shape, random_center=True, random_size=False),
     # mt.RandZoomd(
     #     keys=["image", "mask"],
     #     min_zoom=0.9,
     #     max_zoom=1.2,
     #     mode=("bilinear", "nearest"),
     #     align_corners=(True, None),
     #     prob=1,
     # ),
     # mt.Rand2DElasticD(
     #     keys=["image", "mask"],
     #     prob=1,
     #     spacing=(50, 50),
     #     magnitude_range=(1, 3),
     #     rotate_range=(np.pi / 4,),
     #     scale_range=(0.1, 0.1),
     #     translate_range=(10, 10),
     #     padding_mode="border",
     # ),
     # mt.RandScaleIntensityd(keys=["image"], factors=0.3, prob=1),
     # mt.RandFlipd(["image", "mask"], spatial_axis=[0], prob=1),
     # mt.RandFlipd(["image", "mask"], spatial_axis=[1], prob=1),
     # mt.RandRotateD(keys=["image", "mask"], range_x=np.pi / 4, range_y=np.pi / 4, range_z=0.0, prob=1,
     #                keep_size=True, mode=("nearest", "nearest"), align_corners=False),
     # mt.RandRotate90D(keys=["image", "mask"], prob=1, spatial_axes=(0, 1)),
     # mt.RandGaussianNoiseD(keys=["image"], prob=1, std=0.01),
     mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False),
     # mt.RandKSpaceSpikeNoiseD(keys=["image"], prob=1, intensity_range=(5.0, 7.5)),
     ]
)

val_compose = mt.Compose(
    [
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False),
    ]
)

test_compose = mt.Compose(
    [
        mt.DivisiblePadD(keys=["image", "mask"], k=(16, 16), mode="edge"),
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False),
    ]
)

splits = KFold(n_splits=5, shuffle=True, random_state=4)

concatenated_dataset = train_loader_ACDC(transform=None, train_index=None)

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(concatenated_dataset)))):

    print("--------------------------", "Fold", fold + 1, "--------------------------")

    # training dataset
    training_data = DataLoader(train_loader_ACDC(transform=train_compose, train_index=train_idx), batch_size=5,
                               shuffle=True)
    print("train from here")
    for dic in training_data:
        images = dic["image"]
        masks = dic["mask"]
        # print(images.shape, masks.shape)
        # image, label = dic["image"], dic["mask"]
        # plt.figure("visualise", (8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("image")
        # plt.imshow(image[0, 0, :, :], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.title("mask")
        # plt.imshow(label[0, 0, :, :], cmap="gray")
        # plt.show()

    # validation dataset
    validation_data = DataLoader(val_loader_ACDC(transform=val_compose, val_index=val_idx), batch_size=1,
                                 shuffle=False)
    print("val from here")
    for dic in validation_data:
        images = dic["image"]
        masks = dic["mask"]
        # print(images.shape, masks.shape)
        # image, label = dic["image"], dic["mask"]
        # plt.figure("visualise", (8, 4))
        # plt.subplot(1, 2, 1)
        # plt.title("image")
        # plt.imshow(image[0, 0, :, :], cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.title("mask")
        # plt.imshow(label[0, 0, :, :], cmap="gray")
        # plt.show()

    # test dataset
    test_data = DataLoader(test_loader_ACDC(transform=test_compose, test_index=None), batch_size=1, shuffle=False)
    print("test from here")
    for dic in test_data:
        images = dic["image"]
        masks = dic["mask"]
        print(images.shape, masks.shape)
