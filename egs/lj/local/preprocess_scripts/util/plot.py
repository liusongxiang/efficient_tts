import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_alipred(ref_ali, pred_ali, path):
  f, axarr = plt.subplots(2, sharex=True)
  axarr[0].plot(ref_ali, 'r')
  axarr[0].set_title('Reference per-frame phoneme alignment')
  axarr[1].plot(pred_ali, 'b')
  axarr[1].set_title('RNN-T predicted per-frame phoneme alignment')
  plt.savefig(path, format='png')

def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')

def plot_mel(mel, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    mel,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.savefig(path, format='png')