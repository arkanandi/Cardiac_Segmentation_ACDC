import matplotlib.pyplot as plt

from data_3d import *
from train_3d import *

torch.manual_seed(42)
import monai.transforms as mt

"""-----------------------Arguments-----------------------"""
parser = argparse.ArgumentParser(description="Prediction")
parser.add_argument("--batch_size_test", type=str, default=1)

args = parser.parse_args()
test_batch_size = args.batch_size_test

# Softmax
soft = torch.nn.Softmax(dim=1).cuda()
# IoU
IOU_metric = IoU(num_classes=4, absent_score=-1., reduction="none").cuda()
# F1 score
f1_metric = F1(num_classes=4, mdmc_average="samplewise", average='none').cuda()

"""---------Post Processing---------"""
keep_largest = monai.transforms.KeepLargestConnectedComponent(applied_labels=[0, 1, 2, 3], independent=True)
fill_holes = monai.transforms.FillHoles()

"""---------Test Data---------"""
test_transform = mt.Compose([
    mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
])
# Test dataset
test_data = DataLoader(test_loader_ACDC3(transform=test_transform, test_index=None), batch_size=1, shuffle=False)


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


# pass the padded image, the indices and the original shape
def UnPad_imges(image, indices, org_shape):
    b, c, h, w, d = org_shape
    xx = indices[0]
    yy = indices[1]
    zz = indices[2]
    return image[:, :, xx:xx + h, yy:yy + w, zz:zz + d]  # image is a torch tensor --> have to move to GPU


# def show_results(res):
#     for slices in range(res.shape[2]):
#         out_show_res = res[:, :, slices]
#         plt.imshow(out_show_res)
#         plt.show()


# save the predictions and ground truths
def save_pred(img, mask, pred, outpath, name_model, idx, aff):
    # Folder to save the results
    if not os.path.exists(os.path.join(outpath, name_model)):
        os.makedirs(os.path.join(outpath, name_model))
    out_save_path_image = os.path.join(outpath, name_model, f"{idx}_image" + '.nii.gz')
    out_save_path_pred = os.path.join(outpath, name_model, f"{idx}_pred" + '.nii.gz')
    out_save_path_mask = os.path.join(outpath, name_model, f"{idx}_gt" + '.nii.gz')
    # affine = np.diag([-1.25, -1.25, 10.0, 1.0])
    # print(aff.shape, aff)
    aff = aff.squeeze().cpu()
    affine = np.diag([torch.diagonal(aff)[0], torch.diagonal(aff)[1],
                      torch.diagonal(aff)[2], torch.diagonal(aff)[3]])
    print(affine)

    # Save images
    img = img.squeeze()
    img = np.array(img.cpu())
    # show_results(img)
    # print(type(img))
    for slices in range(img.shape[2]):
        out_show_img = img[:, :, slices]
        if not os.path.exists(os.path.join(outpath, name_model)):
            os.makedirs(os.path.join(outpath, name_model))
        out_save_path_image = os.path.join(outpath, name_model, f"{idx}_{slices}_image" + '.png')
        image_file_name = f"{idx}_{slices}_image"
        plt.title = image_file_name
        plt.imsave(out_save_path_image, out_show_img, format='png', cmap='gray')
        plt.close()
    # saves the resampled images
    img = nib.Nifti1Image(img, affine)
    nib.save(img, out_save_path_image)

    # Save ground truths
    mask = mask.squeeze()
    mask = np.array(mask.cpu())
    # print(type(mask))
    for slices in range(mask.shape[2]):
        out_show_mask = mask[:, :, slices]
        if not os.path.exists(os.path.join(outpath, name_model)):
            os.makedirs(os.path.join(outpath, name_model))
        out_save_path_mask = os.path.join(outpath, name_model, f"{idx}_{slices}_gt" + '.png')
        image_file_name = f"{idx}_{slices}_gt"
        plt.title = image_file_name
        plt.imsave(out_save_path_mask, out_show_mask, format='png', cmap='gray')
        plt.close()
    # saves the resampled masks
    mask = nib.Nifti1Image(mask, affine)
    nib.save(mask, out_save_path_mask)

    # Save predictions
    # Post Processing
    final_prediction = torch.argmax(pred, dim=1)
    final_prediction = keep_largest(final_prediction)
    # final_prediction = fill_holes(final_prediction)
    # final_prediction = torch.argmax(final_prediction, dim=1)
    final_pred = np.array(final_prediction.cpu().squeeze())
    # print(type(final_pred))
    for slices in range(final_pred.shape[2]):
        out_show_pred = final_pred[:, :, slices]
        if not os.path.exists(os.path.join(outpath, name_model)):
            os.makedirs(os.path.join(outpath, name_model))
        out_save_path_pred = os.path.join(outpath, name_model, f"{idx}_{slices}_pred" + '.png')
        image_file_name = f"{idx}_{slices}_pred"
        plt.title = image_file_name
        plt.imsave(out_save_path_pred, out_show_pred, format='png', cmap='gray')
        plt.close()
    # saves the resampled predictions
    final_pred = nib.Nifti1Image(final_pred, affine)
    nib.save(final_pred, out_save_path_pred)


def save_results(iou, dice, out_path, model_name):
    # Folder to store the plots
    if not os.path.exists(os.path.join(out_path, model_name)):
        os.makedirs(os.path.join(out_path, model_name))
    out_save_path = os.path.join(os.path.join(out_path, model_name))
    # IoU
    sorted_iou = sorted(iou)
    print("Top IoU:", sorted_iou[-1])
    # Dice Scores
    sorted_dice = sorted(dice)
    print("Top Dice Score:", sorted_dice[-1])
    # save results
    result_dict = {"IoU": sorted_iou[-1],
                   "Dice Score": sorted_dice[-1]}

    file_name = 'results.txt'
    completeName = os.path.join(out_save_path, file_name)
    with open(completeName, 'w') as file:
        file.write(str(result_dict))


def test_results(model, out_path, model_name):
    all_iou = []
    all_dice = []
    indices = 0
    for items in test_data:
        image = items["image"].cuda()
        image_shape = image.shape
        mask = items["mask"].long().cuda().squeeze(dim=1)
        # print(mask.shape, image_shape)
        # pad the image
        image, ind = Pad_images(image)
        pred = model(image.float().cuda())
        # unpad the images
        pred = UnPad_imges(pred, ind, image_shape).cuda()
        pred = soft(pred)
        # pred = torch.argmax(pred, dim=1)
        ###############################################
        img_affine = items['image_meta_dict']['affine']
        mask_affine = items['mask_meta_dict']['affine']
        image_affine_original = items['image_meta_dict']['original_affine']
        mask_affine_original = items['mask_meta_dict']['original_affine']
        # print(img_affine, image_affine_original)
        ###############################################
        # Save results
        save_pred(image, mask, pred, out_path, model_name, indices, image_affine_original)
        # calculate iou
        iou_all_class = IOU_metric(pred, mask)
        iou_all_class = iou_all_class.cpu().numpy()
        iou_all_class = iou_all_class[iou_all_class != -1.]
        iou = iou_all_class.mean()
        all_iou.append(iou)
        # calculate dice score
        dice_all_class = f1_metric(pred, mask)
        dice_all_class = dice_all_class.cpu().numpy()
        # dice_all_class = dice_all_class[~np.isnan(dice_all_class)]
        dice_all_class = dice_all_class[dice_all_class != -1.]
        dice = dice_all_class.mean()
        all_dice.append(dice)
        indices = indices + 1
    # Save plots
    save_results(all_iou, all_dice, out_path, model_name)


if __name__ == "__main__":
    name = str("UNet3D_Best_0.5_Fold_4")
    model_path = str(Path(r"../unet/cluster_results/best_models", name + ".pt"))
    if not os.path.exists(r'../unet/test_results_3d/'):
        os.makedirs(r'../unet/test_results_3d/')
    result_path = r'../unet/test_results_3d/'
    model = torch.load(model_path)
    with torch.no_grad():
        model.eval().cuda()
        test_results(model, result_path, name)
