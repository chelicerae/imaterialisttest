import os
import cv2
import json
import numpy as np
import pandas as pd
import keras_segmentation
import matplotlib.pyplot as plt

import utils

csv_dir = "data/train.csv"
image_df = utils.get_prepproc_data(csv_dir)
print(image_df)

image_df.apply(utils.get_final_mask_img, args=(['data/masks/']), axis=1)

model = keras_segmentation.models.unet.vgg_unet(n_classes=51 ,  input_height=416, input_width=608)

# model.train( 
# 	train_images =  "data/train/",
# 	train_annotations = "data/masks/",
# 	checkpoints_path = "data/checkpoints/" , epochs=5
# )

out = keras_segmentation.predict.predict_multiple(
	model=model,
	inp_dir="data/train/",
	out_dir="data/"
)
