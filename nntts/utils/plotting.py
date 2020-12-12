import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import os
import logging


def plots(imvs, alphas, mel_preds, mel_gts, step, out_dir, num_plots=4):
    output_dir = f"{out_dir}/images/"
    os.makedirs(output_dir, exist_ok=True)

    imvs = imvs.detach().cpu().numpy()
    alphas = alphas.detach().cpu().numpy()
    mel_preds = mel_preds.detach().cpu().numpy()
    mel_gts = mel_gts.detach().cpu().numpy()
    # logging.info(mel_gts.shape)

    i = 1
    # w, h = plt.figaspect(1.0 / len(imvs))
    # fig = plt.Figure(figsize=(w * 1.3, h * 1.3))

    for imv, alpha, mel_pred, mel_gt in zip(imvs, alphas, mel_preds, mel_gts):
        fig, ax = plt.subplots(4)
        ax[0].plot(range(len(imv)), imv)
        ax[1].imshow(alpha[::-1])
        ax[2].imshow(mel_pred.T)
        ax[3].imshow(mel_gt.T)
        fig.savefig(f"{output_dir}/step{step}_{i}.png")
        i += 1
        if i > 4:
            break
     
def plots2(alphas, mel_preds, mel_gts, step, out_dir, num_plots=4):
    output_dir = f"{out_dir}/images/"
    os.makedirs(output_dir, exist_ok=True)

    alphas = alphas.detach().cpu().numpy()
    mel_preds = mel_preds.detach().cpu().numpy()
    mel_gts = mel_gts.detach().cpu().numpy()

    i = 1
    for alpha, mel_pred, mel_gt in zip(alphas, mel_preds, mel_gts):
        fig, ax = plt.subplots(3)
        ax[0].imshow(alpha[::-1])
        ax[1].imshow(mel_pred.T)
        ax[2].imshow(mel_gt.T)
        fig.savefig(f"{output_dir}/step{step}_{i}.png")
        i += 1
        if i > 4:
            break
