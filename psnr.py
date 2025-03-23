import os
import sys
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import piq
import json
import lpips
import matplotlib.pyplot as plt

def compute_metrics(gt_dir, pred_dir):
    gt_filepaths = glob.glob(os.path.join(gt_dir, "*.png"))
    # gt_filepaths = [fp for fp in gt_filepaths if os.path.isfile(os.path.join(pred_dir, os.path.basename(fp)))]
    eps = 0.01
    gt_filepaths.sort()

    CORRECTION = True

    gt_images = []
    pred_images = []
    log_gt_images = []
    log_pred_images = []
    pred_paths = []

    gt_means = []
    pred_means = []

    step = 1

    if len(glob.glob(os.path.join(pred_dir, "*.png"))) == 0:
        raise ValueError("No predictions found")

    print("Computing log images and average colors...")
    for i in tqdm(range(0, len(gt_filepaths), step)):
        gt_fp = gt_filepaths[i]

        gt_image = cv2.imread(gt_fp).astype(np.float128) / 255
        # E3DGS
        curr_pred_paths = glob.glob(os.path.join(pred_dir, "*" + os.path.basename(gt_fp).split('.', 1)[0] + "_*.npy"))
        # EventNeRF
        # curr_pred_paths = glob.glob(os.path.join(pred_dir, "*" + os.path.basename(gt_fp).split('.', 1)[0] + ".*"))
        if len(curr_pred_paths) != 1:
            print("Not found:", gt_fp)
            continue
        pred_path = curr_pred_paths[0]
        pred_paths.append(pred_path)
        pred_image = cv2.cvtColor(np.load(pred_path), cv2.COLOR_BGR2RGB) / 255
        # pred_image = cv2.imread(pred_path).astype(np.float128) / 255

        log_pred_img = np.log(pred_image**2.2+eps)
        if CORRECTION:
            log_gt_img = np.log(gt_image**2.2+eps)

            log_gt_images.append(log_gt_img)
        
        if np.isnan(log_pred_img).any():
            breakpoint()
        log_pred_images.append(log_pred_img)

        gt_images.append(gt_image)
        pred_images.append(pred_image)

    if CORRECTION:
        log_gt_pixels = np.vstack(log_gt_images).reshape(-1, 3)
        log_pred_pixels = np.vstack(log_pred_images).reshape(-1, 3)

        log_diff = log_gt_pixels - log_pred_pixels

        # correction_color in normal cases
        avg_diff = np.average(log_diff, axis=0)

        avg_error_in_diff = np.average(log_diff - avg_diff, axis=0)
        std_error_in_diff = np.std(log_diff - avg_diff, axis=0)

        mask = (avg_error_in_diff - 2 * std_error_in_diff < log_diff) & (log_diff < avg_error_in_diff + 2 * std_error_in_diff)

        # correction color after excluding extreme values
        correction_color = np.sum(log_diff * mask, axis=0) / np.sum(mask, axis=0)
        correction_color = correction_color[np.newaxis, np.newaxis, :]

        print(avg_diff, correction_color)


    # log_gt_mean_color = np.array([[-3.43814351, -3.39297346, -3.30724037]], dtype=np.float128)
    # log_pred_mean_color = np.array([[-3.35444683, -3.2701331 , -3.15707293]], dtype=np.float128)

    metrics = {
        "psnr": [],
        "lpips": [],
        "ssim": [],
    }

    lpips_instance = lpips.LPIPS("alex")
    functions = {
        "psnr": piq.psnr,
        "lpips": lambda x, y: lpips_instance(x * 2 -1, y *2 -1),
        "ssim": piq.ssim,
    }
    if CORRECTION:
        os.makedirs(os.path.join(os.path.dirname(pred_dir), "corrected"), exist_ok=True)

    print("Computing metrics...")
    for i in tqdm(range(0, len(gt_images))):
        gt_image = gt_images[i]
        # pred_image = cv2.imread(os.path.join(pred_dir, os.path.basename(gt_fp))).astype(np.float128)

        # log_gt_img = np.log(gt_image**2.2+eps)
        # log_pred_img = np.log(pred_image**2.2+eps)
        pred_img = pred_images[i]
        pred_path = pred_paths[i]
        log_pred_img = log_pred_images[i]

        # log_gt_mean_color = np.average(log_gt_img.reshape((-1, 3)), axis=0)[np.newaxis, np.newaxis, :]
        # log_pred_mean_color = np.average(log_pred_img.reshape((-1, 3)), axis=0)[np.newaxis, np.newaxis, :]

        # cv2.imwrite(os.path.join("meh", os.path.basename(gt_filepaths[i])) + "_pred_org.png", np.clip((np.exp(log_pred_img) - eps) ** (1/2.2), 0, 255).astype(np.uint8))

        if CORRECTION:
            log_pred_img = log_pred_img + correction_color

            pred_image = np.maximum(0, (np.exp(log_pred_img) - eps)) ** (1/2.2)
            cv2.imwrite(os.path.join(os.path.dirname(pred_dir), "corrected", os.path.splitext(os.path.basename(pred_path))[0] + '.png'), np.clip(pred_image * 255, 0, 255).astype(np.uint8))

        gt_image, pred_image = gt_image.astype(np.float32), pred_img.astype(np.float32)

        gt_image = torch.tensor(gt_image).permute(2, 0, 1)[None, ...]
        pred_image = torch.tensor(pred_image).permute(2, 0, 1)[None, ...]
        pred_image = torch.clamp(pred_image, 0, 1)

        for name in metrics.keys():
            metrics[name].append(functions[name](pred_image, gt_image).item())

    metrics_summary = dict()

    for name in metrics.keys():
        metric = pd.Series(metrics[name])
        desc = metric.describe()
        metrics_summary[name] = {
            "mean": round(desc['mean'], 4),
            "std": round(desc['std'], 4), 
        }

    print(metrics_summary)
    print("\n\n\n")
    
    return metrics_summary, metrics


if __name__ == "__main__":
    # gt_dir = sys.argv[1]
    # pred_dir = sys.argv[2]

    # metrics = compute_metrics(gt_dir, pred_dir)

    # output_json = {
    #     "gt_filepath": gt_dir,
    #     "metrics": metrics
    # }
    # json_path = (pred_dir[:-1] if pred_dir[-1] in '/\\' else pred_dir) + '.json'

    # with open(json_path, "w") as f:
    #     json.dump(output_json, f, indent=4)

    gt_dir = sys.argv[1]
    pred_dir = sys.argv[2]
    # gt_dir = "/CT/EventSLAM/static00/data/synthetic/Company/test/rgb/"
    # pred_dir = "/CT/EventSLAM/work/gaussian-splatting/trainings/synthetic/"

    if pred_dir[-1] == '/':
        pred_dir = pred_dir[:-1]
    

    # trainings = glob.glob(os.path.join(pred_dir, "Company_variable_events_2_windows_frustum_change*"))
    trainings = [pred_dir]
    # trainings.sort()

    all_metrics = []
    # for training in trainings:
    #     if os.path.exists(os.path.join(training, "test/ours_60000/renders")):
            # print(training)
    metrics, raw_metrics = compute_metrics(gt_dir, pred_dir)
    metrics["training_id"] = pred_dir.replace("/CT/EventSLAM/work/", "")
    all_metrics.append(metrics)

    fig, ax1 = plt.subplots()

    for key in raw_metrics.keys():
        n = len(raw_metrics[key])
        if key == "psnr":
            continue
        else:
            ax1.plot(range(n), raw_metrics[key], color="green" if key == "lpips" else "blue", label=key)
    key = "psnr"
    
    ax1.legend(loc="upper right")
    ax1.set_ylim([0, 1])
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(range(n), raw_metrics[key], color="red", label=key)

    ax2.legend(loc="upper left")
    ax2.set_ylim([14, 26])
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(pred_dir, "../../../", "eval.png"))
    
    import pandas as pd 

    df = pd.json_normalize(all_metrics)

    if os.path.exists("evals.csv"):
        df2 = pd.read_csv("evals.csv")
        df = pd.concat([df2, df], ignore_index=True)
    df.to_csv("evals.csv", index=None)