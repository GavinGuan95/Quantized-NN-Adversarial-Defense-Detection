import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

pickle_folder = "./pickle_result"
plt_folder = "./plot_result"
if not os.path.exists(plt_folder):
  os.makedirs(plt_folder)

for pickle_file in os.listdir(pickle_folder):
  plt_file = pickle_file.replace(".pickle", ".png")
  plt_filepath = os.path.join(plt_folder, plt_file)
  with open(os.path.join(pickle_folder, pickle_file), 'rb') as f:
    result_collection = pickle.load(f)

  plt.figure(figsize=(16.0, 10.0))
  bin_num = 100
  histogram_bins = np.linspace(0, 0.035, bin_num)
  plt_x = histogram_bins[1:bin_num]

  if result_collection['clean_uncertainty_list'] is not None:
    result_clean = np.histogram(result_collection['clean_uncertainty_list'], bins=histogram_bins)
    plt_y_clean = result_clean[0] / np.sum(result_clean[0])
    plt.plot(plt_x, plt_y_clean, "-r", label="clean")

  if result_collection['fgsm_uncertainty_list'] is not None:
    result_fgsm = np.histogram(result_collection['fgsm_uncertainty_list'], bins=histogram_bins)
    plt_y_fgsm = result_fgsm[0] / np.sum(result_fgsm[0])
    plt.plot(plt_x, plt_y_fgsm, label="fgsm")

  if result_collection['jsma_uncertainty_list'] is not None:
    result_jsma = np.histogram(result_collection['jsma_uncertainty_list'], bins=histogram_bins)
    plt_y_jsma = result_jsma[0] / np.sum(result_jsma[0])
    plt.plot(plt_x, plt_y_jsma, label="jsma")

  if result_collection['pgd_uncertainty_list'] is not None:
    result_pgd = np.histogram(result_collection['pgd_uncertainty_list'], bins=histogram_bins)
    plt_y_pgd = result_pgd[0] / np.sum(result_pgd[0])
    plt.plot(plt_x, plt_y_pgd, label="pgd")

  plt.axis([0, 0.035, 0, 1])
  plt.legend(loc='upper right')
  plt.savefig(plt_filepath, bbox_inches='tight')