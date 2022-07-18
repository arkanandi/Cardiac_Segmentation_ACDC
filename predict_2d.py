from data_2d import *
from train_2d import *
torch.manual_seed(42)

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
fill_holes = monai.transforms.FillHoles(applied_labels=[0])

"""---------Test Data---------"""
test_transform = mt.Compose([
    mt.ToTensorD(keys=["image", "mask"], allow_missing_keys=False)
])
# Test dataset
test_data = DataLoader(test_loader_ACDC(transform=test_transform, test_index=None), batch_size=1, shuffle=False)


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


# save the predictions and ground truths
def save_pred(img, mask, pred, outpath, name_model, idx):
    # Folder to save the results
    if not os.path.exists(os.path.join(outpath, name_model)):
        os.makedirs(os.path.join(outpath, name_model))
    out_save_path_image = os.path.join(outpath, name_model, f"{idx}_image" + "." + 'png')
    out_save_path_pred = os.path.join(outpath, name_model, f"{idx}_pred" + "." + 'png')
    out_save_path_mask = os.path.join(outpath, name_model, f"{idx}_gt" + "." + 'png')
    # Save images
    img = img.squeeze()
    img = np.array(img.cpu())
    image_file_name = str(idx) + "_image"
    plt.title = image_file_name
    plt.imsave(out_save_path_image, img, format='png', cmap='gray')
    # Save ground truths
    mask = mask.squeeze()
    mask = np.array(mask.cpu())
    image_file_name = str(idx) + "_gt"
    plt.title = image_file_name
    plt.imsave(out_save_path_mask, mask, format='png', cmap='gray')
    # Save predictions
    # Post Processing
    final_prediction = torch.argmax(pred, dim=1)
    final_prediction = keep_largest(final_prediction)
    final_prediction = fill_holes(final_prediction)
    # final_prediction = torch.argmax(final_prediction, dim=1)
    final_pred = np.array(final_prediction.cpu().squeeze())
    image_file_name = str(idx) + "_pred"
    plt.title = image_file_name
    plt.imsave(out_save_path_pred, final_pred, format='png', cmap='gray')
    plt.close()


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
        # pad the image
        image, ind = Pad_images(image)
        pred = model(image.float().cuda())
        # unpad the images
        pred = UnPad_imges(pred, ind, image_shape).cuda()
        pred = soft(pred)
        # pred = torch.argmax(pred, dim=1)
        # Save results
        save_pred(image, mask, pred, out_path, model_name, indices)
        # calculate iou
        iou_all_class = IOU_metric(pred, mask)
        iou_all_class = iou_all_class.cpu().numpy()
        iou_all_class = iou_all_class[iou_all_class != -1.]
        iou = iou_all_class.mean()
        all_iou.append(iou)
        # calculate dice score
        dice_all_class = f1_metric(pred, mask)
        dice_all_class = dice_all_class.cpu().numpy()
        dice_all_class = dice_all_class[~np.isnan(dice_all_class)]
        dice = dice_all_class.mean()
        all_dice.append(dice)
        indices = indices + 1
    # Save plots
    save_results(all_iou, all_dice, out_path, model_name)


if __name__ == "__main__":
    name = str("UNet2D_Attention_Best_0.3_Fold_1")
    model_path = str(Path(r"../unet/cluster_results/best_models", name + ".pt"))
    if not os.path.exists(r'../unet/test_results_1/'):
        os.makedirs(r'../unet/test_results_1/')
    result_path = r'../unet/test_results_1/'
    model = torch.load(model_path)
    with torch.no_grad():
        model.eval().cuda()
        test_results(model, result_path, name)
