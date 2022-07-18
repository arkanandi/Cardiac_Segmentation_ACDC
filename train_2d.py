import glob
import shutil
import argparse
import monai
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchmetrics.functional import iou, dice_score
from torch import nn
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import datetime
import monai.transforms as mt
from torchmetrics import IoU, F1
from attn_unet_2d import AttU_Net2D
from unet_2d import Unet_2d
from data_2d import train_loader_ACDC, val_loader_ACDC
from monai.losses.dice import DiceLoss, DiceCELoss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as ea
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Manual seeding
torch.manual_seed(42)

"""-----------------------Arguments-----------------------"""
parser = argparse.ArgumentParser(description='Training of UNet2D Segmentation')
parser.add_argument("--model_choice", default="UNet2D_Attention", type=str)
parser.add_argument("--kfolds", default=5, type=int)
parser.add_argument("--Loss_choice", default="dice", type=str)
parser.add_argument("--Batch_size_train", default=10, type=int)
parser.add_argument("--Batch_size_val", default=1, type=int)
parser.add_argument("--lr", default=np.float32(0.0005), type=float)
parser.add_argument("--lr_decay", default=np.float32(0.985), type=float)
parser.add_argument("--gpus", default=1, type=int)
parser.add_argument("--maximum_epochs", default=400, type=int)
parser.add_argument("--patience_early_stop", default=400, type=int)
parser.add_argument('--optimizer_choice', default='adam', type=str)
parser.add_argument('--scheduler_choice', default='plateau', type=str)
parser.add_argument('--dropout_rate', default=0.3, type=float)

"""--------------Models, Hyperparameters, Metrics and Variables--------------"""
arguments = parser.parse_args()
model_choice = arguments.model_choice
k_folds = arguments.kfolds
loss_choice = arguments.Loss_choice
batch_size_train = arguments.Batch_size_train
batch_size_val = arguments.Batch_size_val
learning_rate = arguments.lr
LR_decay_rate = arguments.lr_decay
dev = arguments.gpus
max_epochs = arguments.maximum_epochs
patience = arguments.patience_early_stop
optim_choice = arguments.optimizer_choice
scheduler_choice = arguments.scheduler_choice
drop_rate = arguments.dropout_rate

print("Model Choice:", model_choice,
      "Dropout Rate", drop_rate,
      "K Folds:", k_folds,
      "Loss Choice:", loss_choice,
      "LR Decay Rate:", LR_decay_rate,
      "Device:", dev,
      "Patience Early Stopping:", patience,
      "Optimizer:", optim_choice,
      "Scheduler:", scheduler_choice)

# Model
if model_choice == "UNet2D":
    my_model = Unet_2d(drop=drop_rate).cuda()  # with upsample --> Has more parameters
elif model_choice == "UNet2D_Attention":
    my_model = AttU_Net2D(drop=drop_rate).cuda()  # with Attention --> Has the most parameters
else:
    raise ValueError("Wrong model choice!")
# Loss Function
if loss_choice == "dice_ce":
    loss_func = DiceCELoss(include_background=True,
                           to_onehot_y=True,
                           sigmoid=False,
                           softmax=True,
                           jaccard=False,
                           reduction="mean",
                           smooth_nr=1e-05,
                           smooth_dr=1e-05,
                           # ce_weight=class_weights,
                           batch=False).cuda()
elif loss_choice == "dice":
    loss_func = DiceLoss(include_background=True,
                         to_onehot_y=True,
                         sigmoid=False,
                         softmax=True,
                         jaccard=False,
                         reduction="mean",
                         smooth_nr=1e-05,
                         smooth_dr=1e-05,
                         batch=False).cuda()
else:
    raise ValueError("Wrong loss choice!")

# IoU
IOU_metric = IoU(num_classes=4, absent_score=-1., reduction="none").cuda()
# F1 score
f1_metric = F1(num_classes=4, mdmc_average="samplewise", average='none').cuda()
# Softmax
soft = torch.nn.Softmax(dim=1)
# Required dimensions
tar_shape = [300, 300]
crop_shape = [224, 224]

"""---------Post Processing---------"""
keep_largest = monai.transforms.KeepLargestConnectedComponent(applied_labels=[0, 1, 2, 3])

"""---------Augmentations---------"""

train_transform = mt.Compose(
    [mt.ResizeWithPadOrCropD(keys=["image", "mask"], spatial_size=tar_shape, mode="constant"),
     mt.RandSpatialCropD(keys=["image", "mask"], roi_size=crop_shape, random_center=True, random_size=False),
     mt.Rand2DElasticD(
         keys=["image", "mask"],
         prob=0.25,
         spacing=(50, 50),
         magnitude_range=(1, 3),
         rotate_range=(np.pi / 4,),
         scale_range=(0.1, 0.1),
         translate_range=(10, 10),
         padding_mode="border",
     ),
     # mt.RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
     mt.RandFlipd(["image", "mask"], spatial_axis=[0], prob=0.5),
     mt.RandFlipd(["image", "mask"], spatial_axis=[1], prob=0.5),
     mt.RandRotateD(keys=["image", "mask"], range_x=np.pi / 4, range_y=np.pi / 4, range_z=0.0, prob=0.50,
                    keep_size=True, mode=("nearest", "nearest"), align_corners=False),
     mt.RandRotate90D(keys=["image", "mask"], prob=0.25, spatial_axes=(0, 1)),
     mt.RandGaussianNoiseD(keys=["image"], prob=0.15, std=0.01),
     mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False),
     mt.RandZoomd(
         keys=["image", "mask"],
         min_zoom=0.9,
         max_zoom=1.2,
         mode="nearest",
         align_corners=None,
         prob=0.25,
     ),
     mt.RandKSpaceSpikeNoiseD(keys=["image"], prob=0.15, intensity_range=(5.0, 7.5)),
     ]
)
val_transform = mt.Compose([
    mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
])

"""------------------Datasets and Directories to save the results------------------"""
# Define the K-fold Cross Validator
splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# train + val dataset for 5 fold cross validation training
concatenated_dataset = train_loader_ACDC(transform=None, train_index=None)

#  paths to store the checkpoints
if not os.path.exists(r"../unet/checkpoints"):
    os.makedirs(r"../unet/checkpoints")
checkpoint_path = "../unet/checkpoints"

if not os.path.exists(r"../unet/tb_logs"):
    os.makedirs(r"../unet/tb_logs")
tb_path = "../unet/tb_logs"

if not os.path.exists(r"../unet/csv_logs"):
    os.makedirs(r"../unet/csv_logs")
csv_path = "../unet/csv_logs"

#  Temporarily store the validated image and ground truth plots --> to be moved to the respective folders
if not os.path.exists(r'../unet/val_images_temp_2d/'):
    os.makedirs(r'../unet/val_images_temp_2d/')
val_path = r'../unet/val_images_temp_2d/'

# Save the validation images and ground truths
if not os.path.exists(r'../unet/val_images_save_2d/'):
    os.makedirs(r'../unet/val_images_save_2d/')
image_path = r'../unet/val_images_save_2d/'


# pad the images so that they are divisible by 16
def Pad_images(image):
    orig_shape = list(image.size())
    original_x = orig_shape[2]
    original_y = orig_shape[3]
    new_x = (16 - (original_x % 16)) + original_x
    new_y = (16 - (original_y % 16)) + original_y
    new_shape = [new_x, new_y]
    b, c, h, w = image.shape
    m = image.min()
    x_max = new_shape[0]
    y_max = new_shape[1]
    result = torch.Tensor(b, c, x_max, y_max).fill_(m)
    xx = (x_max - h) // 2
    yy = (y_max - w) // 2
    result[:, :, xx:xx + h, yy:yy + w] = image
    return result, tuple([xx, yy])  # result is a torch tensor in CPU --> have to move to GPU


# pass the padded image, the indices and the original shape
def UnPad_imges(image, indices, org_shape):
    b, c, h, w = org_shape
    xx = indices[0]
    yy = indices[1]
    return image[:, :, xx:xx + h, yy:yy + w]  # image is a torch tensor --> have to move to GPU


# reset the parameters of the model
def reset_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or \
            isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        m.reset_parameters()


# save the masks
def save_plots_mask(target, idx):
    out_path = os.path.join(val_path, f"{idx}_gt" + "." + 'png')
    out_save_path = os.path.join(image_path, f"{idx}_gt" + "." + 'png')
    target = target.squeeze()
    target = np.array(target.cpu())
    plt.imsave(out_save_path, target)
    image_file_name = str(idx) + "_gt"
    plt.title = image_file_name
    plt.imsave(out_path, target, format='png')
    plt.close()


# save the predictions
def save_plots_pred(pred, idx):
    out_path = os.path.join(val_path, f"{idx}_pred" + "." + 'png')
    out_save_path = os.path.join(image_path, f"{idx}_pred" + "." + 'png')
    soft_pred_log = soft(pred)
    final_pred_log = torch.argmax(soft_pred_log, dim=1)
    #  Post Processing after softmax and argmax
    final_pred_log = keep_largest(final_pred_log)
    final_pred_log = np.array(final_pred_log.cpu().squeeze())
    plt.imsave(out_save_path, final_pred_log)
    image_file_name = str(idx) + "_pred"
    plt.title = image_file_name
    plt.imsave(out_path, final_pred_log, format='png')
    plt.close()


# save the images
def save_plots_image(img, idx):
    out_path = os.path.join(val_path, f"{idx}_image" + "." + 'png')
    out_save_path = os.path.join(image_path, f"{idx}_image" + "." + 'png')
    final_image = np.array(img.cpu().squeeze())
    plt.imsave(out_save_path, final_image)
    image_file_name = str(idx) + "_image"
    plt.title = image_file_name
    plt.imsave(out_path, final_image, format='png')
    plt.close()


class Train2D(pl.LightningModule):
    def __init__(self):
        super(Train2D, self).__init__()
        self.net = my_model
        self.loss_function = loss_func

    def forward(self, x):
        return self.net(x)  # returns output of the model --> B Classes H W

    def training_step(self, batch, batch_idx):
        img, mask = batch["image"], batch["mask"]  # image --> torch.float(), mask --> torch.Long
        img = img.float()  # B Channels H W
        mask = mask.long()  # B Channels H W
        # image passed through the model
        out = self(img)  # B Classes H W
        # calculate loss
        loss = self.loss_function(out, mask)
        # calculate softmax of the prediction
        soft_out = soft(out)  # softmax of the prediction
        mask = mask.squeeze(dim=1)  # B H W
        """ Calculation of metrics using Torchmetrics"""
        # # calculate iou
        # iou_all = IOU_metric(soft_out, mask)
        # # train_iou = iou_all.mean()
        # iou_all = iou_all[iou_all != -1.]
        # if len(iou_all) == 0:
        #     train_iou = torch.tensor(0.0).cuda()
        # else:
        #     train_iou = iou_all.mean()
        # # calculate dice score
        # dice_all = f1_metric(soft_out, mask)
        # # train_dice = dice_all.mean()
        # dice_all_np = dice_all.cpu().numpy()
        # dice_all_np = dice_all_np[~np.isnan(dice_all_np)]
        # dice_all = (torch.from_numpy(dice_all_np).cuda())
        # if len(dice_all) == 0:
        #     train_dice = torch.tensor(0.0).cuda()
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
        # logger --> log the train_loss
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
        return {"loss": loss, "train_iou": train_iou, "train_dice": train_dice}

    def validation_step(self, batch, batch_idx):
        img, mask = batch["image"], batch["mask"]  # image --> torch.float(), mask --> torch.Long
        img = img.float()  # B Channels H W
        mask = mask.long()  # B Channels H W
        ###############################################
        save_plots_image(img, batch_idx)  # save the images
        save_plots_mask(mask, batch_idx)  # save the masks
        ###############################################
        # pad the image
        padded_image, ind = Pad_images(img)
        padded_image = padded_image.cuda()
        # padded image passed through the model
        out = self(padded_image).cuda()  # B Classes H W
        # unpad the image
        unpadded_prediction = UnPad_imges(out, ind, img.shape)
        unpadded_prediction = unpadded_prediction.cuda()
        ###############################################
        save_plots_pred(unpadded_prediction, batch_idx)  # save the predictions
        ###############################################
        # calculate loss
        loss = self.loss_function(unpadded_prediction, mask)
        # calculate softmax of the prediction
        soft_out = soft(unpadded_prediction)
        mask = mask.squeeze(dim=1)  # B H W
        """ Calculation of metrics using Torchmetrics"""
        # # calculate iou
        # iou_all = IOU_metric(soft_out, mask)
        # # val_iou = iou_all.mean()
        # iou_all = iou_all[iou_all != -1.]
        # if len(iou_all) == 0:
        #     val_iou = torch.tensor(0.0).cuda()
        # else:
        #     val_iou = iou_all.mean()
        # # calculate dice score
        # dice_all = f1_metric(soft_out, mask)
        # # val_dice = dice_all.mean()
        # dice_all_np = dice_all.cpu().numpy()
        # dice_all_np = dice_all_np[~np.isnan(dice_all_np)]
        # dice_all = (torch.from_numpy(dice_all_np).cuda())
        # if len(dice_all) == 0:
        #     val_dice = torch.tensor(0.0).cuda()
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
        # logger --> log the val_loss
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=False)
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
            scheduler = ReduceLROnPlateau(optim, mode='min', factor=LR_decay_rate, patience=20)
        elif scheduler_choice == 'step':
            scheduler = StepLR(optim, step_size=10, gamma=LR_decay_rate, last_epoch=-1)
        else:
            raise ValueError("Wrong scheduler!")
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            'monitor': 'avg_train_loss'
        }


def run_training():
    """--------------------------------------5 fold Cross Validation--------------------------------------"""
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(concatenated_dataset)))):
        print(len(train_idx), len(val_idx))
        print("--------------------------", "Fold", fold + 1, "--------------------------")
        print("Train Batch Size:", batch_size_train,
              "Val Batch Size:", batch_size_val,
              "Learning Rate:", learning_rate,
              "Max epochs:", max_epochs)

        """-------------------Train the model for "max_epochs" for each fold-------------------"""
        # training dataset
        training_data = DataLoader(train_loader_ACDC(transform=train_transform, train_index=train_idx),
                                   batch_size=batch_size_train,
                                   shuffle=True, num_workers=2)
        # validation dataset
        validation_data = DataLoader(val_loader_ACDC(transform=val_transform, val_index=val_idx),
                                     batch_size=batch_size_val, shuffle=False, num_workers=2)
        # init the model
        model = Train2D()
        # name of the model
        name = str(model_choice) + "_" + str(drop_rate) + "_" + str(datetime.date.today()) + "_Fold_" + str(fold + 1)
        #  Checkpoint callback and Early Stopping
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path,
                                                           save_top_k=1,
                                                           save_last=True,
                                                           verbose=True,
                                                           monitor='avg_val_iou',
                                                           mode='max',
                                                           filename=name + "_" + '{epoch}-{avg_val_iou:.4f}',
                                                           )
        early_stop_callback = pl.callbacks.EarlyStopping(monitor='avg_val_loss',
                                                         min_delta=0.00,
                                                         patience=patience,
                                                         verbose=False,
                                                         mode='min')
        # Tensorboard logger --> tensorboard --logdir=tb_logs
        tensorboard_logger = TensorBoardLogger(tb_path, name=name)
        # CSV logger
        csv_logger = CSVLogger(csv_path, name=name)
        # Trainer for training
        trainer = Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback, checkpoint_callback],
                          gpus=1, logger=[tensorboard_logger, csv_logger], fast_dev_run=False, log_every_n_steps=2)
        # Training the model
        trainer.fit(model, train_dataloader=training_data, val_dataloaders=validation_data)

        # Save the best model
        best_model_path = checkpoint_callback.best_model_path
        model = model.load_from_checkpoint(best_model_path)
        model.eval().cuda()
        fname = str(model_choice) + "_Best_" + str(drop_rate) + "_Fold_" + str(fold + 1)
        if not os.path.exists(r'../unet/best_models/'):
            os.makedirs(r'../unet/best_models/')
        torch.save(model, str(Path('../unet/best_models/', fname + '.pt')))

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

        # reset model parameters after each fold
        model.apply(reset_weights)


if __name__ == "__main__":
    run_training()
