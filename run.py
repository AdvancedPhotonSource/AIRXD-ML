import os
from glob import glob
from src.utilities import Dataset, parse_imctrl
from src.mask import *
from src.airxd import ARIXD
from sklearn.metrics import confusion_matrix as CM

import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = Dataset()
images = dataset.get_images('data')
labels = np.zeros_like(images)

# get labels / masks
controls = parse_imctrl('data/Si_ch3_d700-00000.imctrl')
mask = MASK(controls=controls, shape=(2880,2880))
if os.path.isdir('data/masks'):
    lpath = sorted(glob('data/masks/*.npy'))
    for i, path in enumerate(lpath):
        label = np.load(path)
        labels[i] += label
else:
    os.mkdir('data/masks')
    for i, image in enumerate(images):
        result = mask.AutoSpotMask(image, esdmul=7.0)
        labels[i] += result
        np.save(f'data/masks/{i}', result)

# Train
device = 'cpu'
tree_method = 'hist' if device == 'cpu' else 'gpu_hist'
predictor = 'cpu_predictor' if device == 'cpu' else 'gpu_predictor'
max_depth = 10

algorithm = 'gradient_boosting'
parameters = {'n_estimators': 100,
              'eta': 0.3,
              'tree_method': tree_method,
              'gamma': 0.,
              'max_bin': 10000,
              'predictor': predictor,
              'max_depth': max_depth,
              'reg_lambda': 1.,
              'eval_metric': 'rmsle',
              'n_jobs': -1}
features = {'intensity': True,
            'angle': True,
            'locations': True}

model = ARIXD(algorithm, parameters, features, mask.TA)
model.train(images, labels)

# Prediction
for image, label in zip(images, labels):
    label_pred = model.predict(image)
    matrix = CM(label.ravel(), label_pred.ravel())
    tn, fp, fn, tp = matrix.ravel()
    tn_rate = tn/(fp+tn)*100
    tp_rate = tp/(fn+tp)*100
    print(f'True Negative   : {tn}')
    print(f'False Positive  : {fp}')
    print(f'False Negative  : {fn}')
    print(f'True Positive   : {tp}')
    print(f'True TN rate    : {round(tn_rate,1)} %')
    print(f'True TP rate    : {round(tp_rate,1)} %')
