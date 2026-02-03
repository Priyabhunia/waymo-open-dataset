# Viewing Waymo Open Dataset TFRecords from Google Drive in Colab

This guide shows a minimal, reliable path to load TFRecord data stored in
Google Drive and preview camera images in Google Colab. It focuses on using the
published pip package and the existing tutorial code paths so you do not need
to build the package from source.

## 1) Install the Waymo Open Dataset package

Use the TensorFlow 2.12 build to match the current tutorials:

```bash
pip install waymo-open-dataset-tf-2-12-0==1.6.7
pip install Pillow==9.2.0
```

## 2) Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3) Point the tutorial loader at your TFRecord

Replace the tutorial `FILENAME` with your Drive path. If your TFRecord is gzip
compressed, set `compression_type='GZIP'`.

```python
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

FILENAME = '/content/drive/MyDrive/waymo/segment-000.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break
```

## 4) Preview a camera image from the first frame

```python
import numpy as np
from PIL import Image
from waymo_open_dataset.utils import frame_utils

images = sorted(frame.images, key=lambda i: i.name)
img = Image.fromarray(np.frombuffer(images[0].image, dtype=np.uint8))
display(img)
```

## Common issues and fixes

### Wrong TFRecord path or missing permissions

If Colab reports "file not found", verify the Drive path in `FILENAME`. The
Drive root is `/content/drive/MyDrive/`.

### `DataLossError` or "corrupt record"

This typically means the compression type is wrong. Try:

```python
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='GZIP')
```

### Import errors or missing symbols

Use the pip package for your TensorFlow version. The tutorials use
`waymo-open-dataset-tf-2-12-0==1.6.7`.
