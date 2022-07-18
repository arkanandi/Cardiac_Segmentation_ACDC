import os
import glob
import monai
import torch
import shutil
import argparse
import warnings
import datetime
import numpy as np
from torch import nn
import nibabel as nib
from pathlib import Path
from unet_3d import Unet_3d
import monai.transforms as mt
import pytorch_lightning as pl
from torchmetrics import IoU, F1
from attn_unet_3d import Attn_UNet3d
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchmetrics.functional import dice_score, iou
from data_3d import train_loader_ACDC, val_loader_ACDC
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as ea

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
torch.manual_seed(12345)

"""-----------------------Arguments-----------------------"""
parser = argparse.ArgumentParser(description='Training of UNet3d-Segmentation')
parser.add_argument("--model_choice", default="UNet3D_Attention", type=str)
parser.add_argument("--kfolds", default=5, type=int)
parser.add_argument("--Batch_size_train", default=2, type=int)
parser.add_argument("--Batch_size_val", default=1, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("--lr_decay", default=0.980, type=float)
parser.add_argument("--gpus", default=1, type=int)
parser.add_argument("--maximum_epochs", default=400, type=int)
parser.add_argument("--patience_early_stop", default=400, type=int)
parser.add_argument('--monitor', default='avg_val_iou', type=str)
parser.add_argument('--Monitor_mode', default='max', type=str)
parser.add_argument('--optimizer_choice', default='adam', type=str)
parser.add_argument('--scheduler_choice', default='plateau', type=str)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--scheduler_patience', default=10, type=int)

"""--------------Models, Hyperparameters, Metrics and Variables--------------"""
arguments = parser.parse_args()
model_choice = arguments.model_choice
k_folds = arguments.kfolds
batch_size_train = arguments.Batch_size_train
batch_size_val = arguments.Batch_size_val
learning_rate = arguments.lr
LR_decay_rate = arguments.lr_decay
dev = arguments.gpus
max_epochs = arguments.maximum_epochs
patience = arguments.patience_early_stop
monitor_choice = arguments.monitor
monitor_mode = arguments.Monitor_mode
optim_choice = arguments.optimizer_choice
scheduler_choice = arguments.scheduler_choice
scheduler_patience = arguments.scheduler_patience
drop_rate = arguments.dropout_rate

print("Model Choice:", model_choice,
      "Dropout Rate", drop_rate,
      "K Folds:", k_folds,
      "LR Decay Rate:", LR_decay_rate,
      "Device:", dev,
      "Patience Early Stopping:", patience,
      "Monitor Choice to save the model:", monitor_choice,
      "Monitor Mode(max/min) to save the model:", monitor_mode,
      "Optimizer:", optim_choice,
      "Scheduler:", scheduler_choice,
      "Scheduler patience:", scheduler_patience)

# model
if model_choice == "UNet3D":
    my_model = Unet_3d(drop=drop_rate).cuda()  # without attention
elif model_choice == "UNet3D_Attention":
    my_model = Attn_UNet3d(drop=drop_rate).cuda()  # with attention
else:
    raise ValueError("Wrong model choice!")

#  IoU
IOU_metric = IoU(num_classes=4, absent_score=-1., reduction='none').cuda()

# F1 score
f1_metric = F1(num_classes=4, mdmc_average='samplewise', average='none').cuda()

# softmax
soft = torch.nn.Softmax(dim=1)

# target/crop shape for the images and masks when training
tar_shape = [300, 300, 18]
crop_shape = [224, 224, 10]

"""---------Augmentations---------"""
train_compose = mt.Compose(
    [
        mt.ResizeWithPadOrCropD(keys=["image", "mask"], spatial_size=tar_shape),
        mt.RandSpatialCropD(keys=["image", "mask"], roi_size=crop_shape, random_center=True, random_size=False),
        mt.RandRotate90D(keys=["image", "mask"], prob=0.5, spatial_axes=(0, 1)),
        mt.RandAxisFlipD(keys=["image", "mask"], prob=0.5),
        mt.RandKSpaceSpikeNoiseD(keys=["image"], intensity_range=(5.0, 7.5), prob=0.15),
        mt.RandGaussianNoiseD(keys=["image"], mean=0.0, std=0.2, prob=0.25),
        # mt.RandAffineD(keys=["image", "mask"], prob=0.15, rotate_range=(0, 0, 2), translate_range=(0, 0, 2),
        #                scale_range=(0, 0, 2), mode="nearest"),
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)

    ]
)
val_compose = mt.Compose(
    [
        mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False),
    ]
)

"""------------------Datasets and Directories to save the results------------------"""
# Define the K-fold Cross Validator
splits = KFold(n_splits=k_folds, shuffle=True, random_state=12345)

# train + val dataset for 5 fold cross validation training
concatenated_dataset = train_loader_ACDC(transform=None, train_index=None)

#  path to store the checkpoints and the best model
if not os.path.exists("../unet/checkpoints"):
    os.mkdir("../unet/checkpoints")
checkpoint_path = "../unet/checkpoints"

if not os.path.exists("../unet/tb_logs"):
    os.mkdir("../unet/tb_logs")
tb_path = "../unet/tb_logs"

if not os.path.exists("../unet/csv_logs"):
    os.mkdir("../unet/csv_logs")
csv_path = "../unet/csv_logs"

# Temporarily store the validated image and ground truth plots --> to be moved to the respective folders
if not os.path.exists(r'../unet/val_images_temp_3d/'):
    os.makedirs(r'../unet/val_images_temp_3d/')
val_path = r'../unet/val_images_temp_3d/'

# Save the validation images and ground truths
if not os.path.exists(r'../unet/val_images_save_3d/'):
    os.makedirs(r'../unet/val_images_save_3d/')
image_path = r'../unet/val_images_save_3d/'

"""---------Post Processing---------"""
keep_largest = monai.transforms.KeepLargestConnectedComponent(applied_labels=[1, 2, 3])


#  padding: just pass the image
def Pad_images(image):
    orig_shape = list(image.size())
    original_x = orig_shape[2]
    original_y = orig_shape[3]
    original_z = orig_shape[4]
    new_x = (16 - (original_x % 16)) + original_x
    new_y = (16 - (original_y % 16)) + original_y
    new_z = original_z
    new_shape = [new_x, new_y, new_z]
    b, c, h, w, d = image.shape
    m = image.min()
    x_max = new_shape[0]
    y_max = new_shape[1]
    z_max = new_shape[2]
    result = torch.Tensor(b, c, x_max, y_max, z_max).fill_(m)
    xx = (x_max - h) // 2
    yy = (y_max - w) // 2
    zz = (z_max - d) // 2
    result[:, :, xx:xx + h, yy:yy + w, zz:zz + d] = image
    return result, tuple([xx, yy, zz])  # result is a torch tensor in CPU --> have to move to GPU


#  pass the padded image, the indices and the original shape
def UnPad_imges(image, indices, org_shape):
    b, c, h, w, d = org_shape
    xx = indices[0]
    yy = indices[1]
    zz = indices[2]
    return image[:, :, xx:xx + h, yy:yy + w, zz:zz + d]  # image is a torch tensor --> have to move to GPU


"""-----------------------------------------------------------------------------------"""


# save the images
def save_plots_image(img, idx, img_aff, img_aff_org):
    out_path = os.path.join(val_path, f"{idx}_image" + '.nii.gz')
    final_image = np.array(img.cpu())
    final_image = np.squeeze(final_image)
    img_aff = img_aff.squeeze().cpu()
    affine = np.diag([torch.diagonal(img_aff)[0], torch.diagonal(img_aff)[1],
                      torch.diagonal(img_aff)[2], torch.diagonal(img_aff)[3]])
    final_image = nib.Nifti2Image(final_image, affine=affine)
    nib.save(final_image, out_path)


# save the masks
def save_plots_mask(target, idx, gt_aff, gt_aff_org):
    out_path = os.path.join(val_path, f"{idx}_mask" + '.nii.gz')
    final_mask = np.array(target.cpu())
    final_mask = np.squeeze(final_mask)
    gt_aff = gt_aff.squeeze().cpu()
    affine = np.diag([torch.diagonal(gt_aff)[0], torch.diagonal(gt_aff)[1],
                      torch.diagonal(gt_aff)[2], torch.diagonal(gt_aff)[3]])
    final_mask = nib.Nifti2Image(final_mask, affine=affine)
    nib.save(final_mask, out_path)


# save the predictions
def save_plots_pred(pred, idx, pred_aff, pred_aff_org):
    out_path = os.path.join(val_path, f"{idx}_pred" + '.nii.gz')
    soft_pred_log = soft(pred)
    final_pred_log = torch.argmax(soft_pred_log, dim=1)
    # Post Processing after softmax and argmax
    final_pred_log = keep_largest(final_pred_log)
    ####################################################
    final_pred_log = np.array(final_pred_log.cpu())
    final_pred_log = np.squeeze(final_pred_log)
    pred_aff = pred_aff.squeeze().cpu()
    affine = np.diag([torch.diagonal(pred_aff)[0], torch.diagonal(pred_aff)[1],
                      torch.diagonal(pred_aff)[2], torch.diagonal(pred_aff)[3]])
    final_pred_log = nib.Nifti2Image(final_pred_log, affine=affine)
    nib.save(final_pred_log, out_path)


"""-----------------------------------------------------------------------------------"""


class Train_3D(pl.LightningModule):

    def __init__(self):
        super(Train_3D, self).__init__()
        self.net = my_model
        self.loss_function = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch["image"], batch["mask"]  # image --> torch.float(), mask --> torch.Long
        img = img.float()
        mask = mask.long()
        mask = mask.squeeze(dim=1)
        # image passed through the model
        out = self(img)
        # loss
        loss = self.loss_function(out, mask)
        soft_out = soft(out)
        """ Calculation of metrics using Torchmetrics"""
        # # iou
        # iou_all = IOU_metric(soft_out, mask)
        # iou_all = iou_all[iou_all != -1.]
        # if len(iou_all) == 0:
        #     train_iou = 0.0
        # else:
        #     train_iou = iou_all.mean()
        # # dice score
        # dice_all = f1_metric(soft_out, mask)
        # dice_all = dice_all[dice_all != torch.isnan(dice_all)]
        # if len(dice_all) == 0:
        #     train_dice = 0.0
        # else:
        #     train_dice = dice_all.mean()
        """ Calculation of metrics using Torchmetrics functional"""
        # iou
        iou_all = iou(soft_out, mask, absent_score=-1., num_classes=4, reduction='none', ignore_index=None)
        iou_all = iou_all[iou_all != -1.]
        if len(iou_all) == 0:
            train_iou = torch.tensor(0.0).cuda()
        else:
            train_iou = iou_all.mean()
        # dice score
        dice_all = dice_score(soft_out, mask, bg=True, no_fg_score=-1., reduction='none')
        dice_all = dice_all[dice_all != -1.]
        if len(dice_all) == 0:
            train_dice = torch.tensor(0.0).cuda()
        else:
            train_dice = dice_all.mean()
        # logger
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss, "train_iou": train_iou, "train_dice": train_dice}

    def validation_step(self, batch, batch_idx):
        img, mask = batch["image"], batch["mask"]  # image --> torch.float(), mask --> torch.Long
        img = img.float()
        mask = mask.long()
        ###############################################
        img_affine = batch['image_meta_dict']['affine']
        mask_affine = batch['mask_meta_dict']['affine']
        image_affine_original = batch['image_meta_dict']['original_affine']
        mask_affine_original = batch['mask_meta_dict']['original_affine']
        ###############################################
        save_plots_image(img, batch_idx, img_affine, image_affine_original)  # save the images
        save_plots_mask(mask, batch_idx, mask_affine, mask_affine_original)  # save the masks
        ###############################################
        mask = mask.squeeze(dim=1)
        # pad the image
        padded_image, ind = Pad_images(img)
        padded_image = padded_image.cuda()
        # image passed through the model
        out = self(padded_image).cuda()
        # unpad the image
        unpadded_prediction = UnPad_imges(out, ind, img.shape)
        unpadded_prediction = unpadded_prediction.cuda()
        ###############################################
        save_plots_pred(unpadded_prediction, batch_idx, img_affine, image_affine_original)  # save the predictions
        ###############################################
        # loss
        loss = self.loss_function(unpadded_prediction, mask)
        # softmax
        soft_out = soft(unpadded_prediction)
        """ Calculation of metrics using Torchmetrics"""
        # # iou
        # iou_all = IOU_metric(soft_out, mask)
        # iou_all = iou_all[iou_all != -1.]
        # if len(iou_all) == 0:
        #     val_iou = 0.0
        # else:
        #     val_iou = iou_all.mean()
        # # dice score
        # dice_all = f1_metric(soft_out, mask)
        # dice_all = dice_all[dice_all != torch.isnan(dice_all)]
        # if len(dice_all) == 0:
        #     val_dice = 0.0
        # else:
        #     val_dice = dice_all.mean()
        """ Calculation of metrics using Torchmetrics functional"""
        # iou
        iou_all = iou(soft_out, mask, absent_score=-1., num_classes=4, reduction='none', ignore_index=None)
        iou_all = iou_all[iou_all != -1.]
        if len(iou_all) == 0:
            val_iou = torch.tensor(0.0).cuda()
        else:
            val_iou = iou_all.mean()
        # dice score
        dice_all = dice_score(soft_out, mask, bg=True, no_fg_score=-1., reduction='none')
        dice_all = dice_all[dice_all != -1.]
        if len(dice_all) == 0:
            val_dice = torch.tensor(0.0).cuda()
        else:
            val_dice = dice_all.mean()
        # logger
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss, 'val_iou': val_iou, 'val_dice': val_dice}

    def training_epoch_end(self, train_step_outputs):
        """-----Calculate and logs the average train loss, IoU score and Dice Score-----"""
        avg_train_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
        avg_train_iou = torch.stack([x['train_iou'] for x in train_step_outputs]).mean()
        avg_train_dice = torch.stack([x['train_dice'] for x in train_step_outputs]).mean()
        self.log('avg_train_loss', avg_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_train_iou', avg_train_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_train_dice', avg_train_dice, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, val_step_outputs):
        """-----Calculate and logs the average validation loss, IoU score and Dice Score-----"""
        avg_val_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean()
        avg_val_iou = torch.stack([x['val_iou'] for x in val_step_outputs]).mean()
        avg_val_dice = torch.stack([x['val_dice'] for x in val_step_outputs]).mean()
        self.log('avg_val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_val_iou', avg_val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('avg_val_dice', avg_val_dice, on_step=False, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': avg_val_loss, 'avg_val_iou': avg_val_iou, 'avg_val_dice': avg_val_dice}

    def configure_optimizers(self):
        """-----Optimizers and LR Schedulers-----"""
        if optim_choice == 'adam':
            optim = torch.optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-5, amsgrad=True)
        else:
            raise ValueError("Wrong optimizer!")
        if scheduler_choice == 'plateau':
            scheduler = ReduceLROnPlateau(optim, mode='min', factor=LR_decay_rate, patience=scheduler_patience)
        elif scheduler_choice == 'step':
            scheduler = StepLR(optim, step_size=20, gamma=LR_decay_rate)
        else:
            raise ValueError("Wrong scheduler!")
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            'monitor': 'avg_train_loss'
        }


#   Reset the parameters of the model for the next fold
def reset_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        m.reset_parameters()


def run_training():
    """ 5 fold Cross Validation"""
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(concatenated_dataset)))):
        print("--------------------------", "Fold", fold + 1, "--------------------------")
        print("Train Batch Size:", batch_size_train,
              "Val Batch Size:", batch_size_val,
              "Learning Rate:", learning_rate,
              "Max epochs:", max_epochs)

        """-------------------Train the model for "max_epochs" for each fold-------------------"""
        # training dataset
        training_data = DataLoader(train_loader_ACDC(transform=train_compose, train_index=train_idx),
                                   batch_size=batch_size_train,
                                   shuffle=True)
        # validation dataset
        validation_data = DataLoader(val_loader_ACDC(val_index=val_idx, transform=val_compose),
                                     batch_size=batch_size_val,
                                     shuffle=False)
        # init the model
        model = Train_3D()
        # name of the model
        name = str(model_choice) + "_" + str(drop_rate) + "_" + str(datetime.date.today()) + "_Fold_" + str(fold + 1)
        #  Checkpoint callback and Early Stopping
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path,
                                                           save_top_k=1,
                                                           save_last=True,
                                                           verbose=True,
                                                           monitor='avg_val_iou',
                                                           mode='max',
                                                           filename=name + "_" + '{epoch}-{avg_val_iou:.3f}',
                                                           )
        early_stop_callback = pl.callbacks.EarlyStopping(monitor='avg_val_loss',
                                                         min_delta=0.00,
                                                         patience=patience,
                                                         verbose=False,
                                                         mode='min')
        # tensorboard --logdir .
        tensorboard_logger = TensorBoardLogger(tb_path, name=name)
        # CSV logger
        csv_logger = CSVLogger(csv_path, name=name)
        # Trainer for training
        trainer = Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback, checkpoint_callback],
                          gpus=dev, logger=[tensorboard_logger, csv_logger],
                          fast_dev_run=False, log_every_n_steps=2)
        # Training the model
        trainer.fit(model, train_dataloader=training_data, val_dataloaders=validation_data)
        # Save the best model
        file_name = str(model_choice) + "_Best_" + str(drop_rate) + "_Fold_" + str(fold + 1)
        best_model_path = checkpoint_callback.best_model_path
        model = model.load_from_checkpoint(best_model_path)
        model.eval().cuda()
        if not os.path.exists(r'../unet/best_models/'):
            os.makedirs(r'../unet/best_models/')
        torch.save(model, str(Path('../unet/best_models/', file_name + '.pt')))

        # Folders to save the validation images for each fold
        if not os.path.exists(os.path.join(r'../unet/', name, f"{fold + 1}_Fold")):
            os.makedirs(os.path.join(r'../unet/', name, f"{fold + 1}_Fold"))
        val_images_path = os.path.join(r'../unet/', name, f"{fold + 1}_Fold")

        #  Move the validated images to the respective folders
        for filename in glob.glob(os.path.join(val_path, '*.*')):
            shutil.move(filename, val_images_path)

        # Save plots --> Loss, IoU and Dice
        plot_out_path = str(Path(r"../unet/Plots/", name))
        if not os.path.exists(plot_out_path):
            os.makedirs(plot_out_path)

        event_acc = ea(str(Path(r"../unet/tb_logs/", name, "version_0")))
        event_acc.Reload()

        _, _, training_loss = zip(*event_acc.Scalars('avg_train_loss'))
        _, _, validation_loss = zip(*event_acc.Scalars('avg_val_loss'))
        _, _, training_iou = zip(*event_acc.Scalars('avg_train_iou'))
        _, _, validation_iou = zip(*event_acc.Scalars('avg_val_iou'))
        _, _, training_dice = zip(*event_acc.Scalars('avg_train_dice'))
        _, _, validation_dice = zip(*event_acc.Scalars('avg_val_dice'))

        t_loss, v_loss, t_iou, v_iou, t_dice, v_dice = np.array(training_loss), np.array(validation_loss), \
                                                       np.array(training_iou), np.array(validation_iou), \
                                                       np.array(training_dice), np.array(validation_dice)
        min_length = min(len(t_loss), len(v_loss), len(t_iou), len(v_iou), len(t_dice), len(v_dice))
        total_epochs = np.arange(1, min_length + 1)

        # Save the Loss, IoU and Dice plots
        plt.figure(1)
        plt.rcParams.update({'font.size': 15})
        plt.plot(total_epochs, t_loss[0:min_length], 'X-', label='Training Loss', linewidth=2.0)
        plt.plot(total_epochs, v_loss[0:min_length], 'o-', label='Validation Loss', linewidth=2.0)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(str(Path(plot_out_path, 'Loss_Plot.png')), bbox_inches='tight', format='png', dpi=300)
        plt.close()  # Always plt.close() to save memory

        plt.figure(2)
        plt.rcParams.update({'font.size': 15})
        plt.plot(total_epochs, t_iou[0:min_length], 'X-', label='Training IOU', linewidth=2.0)
        plt.plot(total_epochs, v_iou[0:min_length], 'o-', label='Validation IOU', linewidth=2.0)
        plt.xlabel('Epoch')
        plt.ylabel('IOU')
        plt.legend(loc='lower right')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(str(Path(plot_out_path, 'Iou_Plot.png')), bbox_inches='tight', format='png', dpi=300)
        plt.close()  # Always plt.close() to save memory

        plt.figure(3)
        plt.rcParams.update({'font.size': 15})
        plt.plot(total_epochs, t_dice[0:min_length], 'X-', label='Training Dice', linewidth=2.0)
        plt.plot(total_epochs, v_dice[0:min_length], 'o-', label='Validation Dice', linewidth=2.0)
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.legend(loc='lower right')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(str(Path(plot_out_path, 'Dice_Plot.png')), bbox_inches='tight', format='png', dpi=300)
        plt.close()  # Always plt.close() to save memory

        # reset parameters for the next fold
        model.apply(reset_weights)


if __name__ == "__main__":
    run_training()
