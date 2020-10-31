import pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt


# with open('test_pickle.p','rb') as f:
#   result_collection = pickle.load(f)



# histogram_bins = np.linspace(0, 0.35, 100)
#
# np.random.seed(1234)
# rdn_numbers = np.random.rand(1000)
#
# result = np.histogram(rdn_numbers, bins=histogram_bins)
# print(result)
#
# plt_x = histogram_bins[0:99]
# plt_y = result[0]/np.sum(result[0])
#
# plt.plot(plt_x, plt_y)
# plt.axis([0, 0.35, 0, 1])
# plt.show()
#
with open('/home/guanyush/Pictures/ECE1784/Galloway/cleverhans_tutorials/pickle_result/binary_False_rand_False_dropout_0.5_attack_1_2_3_adv_tr_1_bit_0_eps_2.0.p', 'rb') as f:
  result_collection = pickle.load(f)

print("end")


