import os
import monai.transforms as mt
import numpy as np
from pathlib import Path
from torch.utils.data.dataset import Dataset


def normalize(data):
    data = (data - data.mean()) / data.std()
    return data


class ACDC_3D(Dataset):
    def __init__(self, source, ind, Transform=None):
        # basic transforms
        self.loader = mt.LoadImaged(keys=["image", "mask"])
        self.add_channel = mt.AddChanneld(keys=["image", "mask"])
        self.spatial_pad = mt.SpatialPadD(keys=["image", "mask"], spatial_size=tar_shape)
        self.spacing = mt.Spacingd(keys=["image", "mask"], pixdim=(1.25, 1.25, 10.0), mode=("nearest", "nearest"))
        # index
        self.ind = ind
        # transform
        if Transform is not None:
            self.transform = Transform
        else:
            self.transform = mt.Compose([
                mt.SpatialPadD(keys=["image", "mask"], spatial_size=tar_shape),
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
        data_return = self.transform(data_return)
        return data_return


# target/crop shape for the images and masks when training
tar_shape = [300, 300, 10]
crop_shape = [224, 224, 10]


def train_loader_ACDC(train_index, data_path="../training/training", transform=None):
    train_loader = ACDC_3D(source=data_path, Transform=transform, ind=train_index)
    return train_loader


def val_loader_ACDC(val_index, data_path=r"../training/training", transform=None):
    val_loader = ACDC_3D(source=data_path, Transform=transform, ind=val_index)
    return val_loader
