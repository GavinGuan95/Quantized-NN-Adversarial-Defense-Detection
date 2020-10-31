import pickle
from collections import OrderedDict
import numpy as np

binary = True

result_collection = OrderedDict([
  ('binary', binary),
  ('clean_accuracy', None),
  ('clean_group_accuracy', None),
  ('clean_uncertainty_list', None),
  ('clean_uncertainty_mean', None),
  ('clean_uncertainty_var', None),
  ('fgsm_accuracy', None),
  ('fgsm_group_accuracy', None),
  ('fgsm_uncertainty_list', None),
  ('fgsm_uncertainty_mean', None),
  ('fgsm_uncertainty_var', None),
  ('fgsm_lr', None),
  ('jsma_accuracy', None),
  ('jsma_group_accuracy', None),
  ('jsma_uncertainty_list', None),
  ('jsma_uncertainty_mean', None),
  ('jsma_uncertainty_var', None),
  ('jsma_lr', None),
  ('pgd_accuracy', None),
  ('pgd_group_accuracy', None),
  ('pgd_uncertainty_list', None),
  ('pgd_uncertainty_mean', None),
  ('pgd_uncertainty_var', None),
  ('pgd_lr', None),
])

result_collection['clean_uncertainty_list'] = np.array([1.0, 2.0, 3.0])
result_collection['clean_uncertainty_var'] = 0.0015
with open('test_pickle.p','wb') as f:
  pickle.dump(result_collection, f)