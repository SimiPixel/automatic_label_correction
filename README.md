# Automatic Label Correction

## Installation
```
pip install git+https://github.com/SimiPixel/automatic_label_correction.git
```

## Usage
```python
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

from ALC import NearestNeighbourCorrection as NNC  
nnc = NNC()

y_corrected = nnc.fit_transform(X, y)
```
### Artifically falsify labels of Iris dataset. Correction factor represents how many false labels are corrected
  - Correction factor = 1: All labels are corrected
  - Correction factor = 0: Same number of incorrect labels than before applying any correction alg.


<p align="center">
<img src="https://github.com/SimiPixel/automatic_label_correction/blob/master/readme_plot.svg" width="1100">
</p>

## API
### Correction algorithms
- ALC.NearestNeighbourCorrection
- ALC.ClusterCorrection
- ALC.AutomaticDataEnhancement
- ALC.BinaryClusterCorrection

- ALC.utils
  - ALC.utils.falsify: Artifically falsify labels
  - ALC.utils.kfold
  - ALC.utils.OneHot
  - ALC.utils.convert_labels: Convert labels into different representation

- ALC.evaluate
  - ALC.evaluate.correction_factor
  - ALC.evaluate.accuracy
