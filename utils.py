import os
import cv2
import json
import numpy as np
import pandas as pd
import keras_segmentation


train_img_dir = 'data/train'
category_num = 27 + 1

def get_prepproc_data(csv_dir):
	"""
	Transform original csv into the pandas dataframe of appropriate format:
	image name, list of encoded segmentations, classes of the segmentations in the same order, 
	height of an image and it's width.

	csv_dir: string
		The directory to original csv file.
	"""

	# Load .csv file to dataframe as it is and add column with list of categories.
	segment_df = pd.read_csv(csv_dir)
	segment_df['CategoryId'] = segment_df['ClassId'].str.split('_').str[0]

	# Group classes and encoded pixels by unique images.  
	image_df = segment_df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
	size_df = segment_df.groupby('ImageId')['Height', 'Width'].mean()
	image_df = image_df.join(size_df, on='ImageId')

	present_img = os.listdir(train_img_dir)
	image_df = image_df[image_df.index.isin(present_img)]

	return image_df.reset_index()


def make_2d_mask(df_row):
	"""
	Transforms the original format of segmentation encoding into 2d matrix where
	each cell corresponds to image pixel and contains the id of class. 

	df_row: pandas.Series
		A row of a dataframe. Must contain the next fields:
		ImageId, EncodedPixels, CategoryId, Height, Width
	"""
	seg_width = df_row.Width
	seg_height = df_row.Height

	# Create 1d array filled with default category id. 
	seg_img = np.full(seg_width * seg_height, category_num-1, dtype=np.int32)

	# Iterate through pairs of encoded csegment and it's category id.
	for encoded_pixels, category in zip(df_row.EncodedPixels, 
										df_row.CategoryId):
		pixels = list(map(int, encoded_pixels.split(" ")))
		for i in range(0, len(pixels), 2):
			start_index = pixels[i] - 1
			index_len = pixels[i+1] - 1
			
			# Fill the segment regions with their id.
			seg_img[start_index:start_index+index_len] = int(category)

	# Reshape 1d array into matrix.
	seg_img = seg_img.reshape((seg_height, seg_width), order='F')
	return seg_img


def get_final_mask_img(df_row, save_dir): 
	"""
	Transforms the segmentation matrix to 3d tensor and saves it to .png.

	df_row: pandas.Series
		A row of a dataframe. Must contain the next fields:
		ImageId, EncodedPixels, CategoryId, Height, Width

	save_dir: string
		The directory of the foder where the mask images will be stored. 
	"""
	mask_2d = make_2d_mask(df_row)
	img_name = df_row.ImageId.split('.')[0] + '.png'
	res = np.dstack((mask_2d, mask_2d, mask_2d))
	cv2.imwrite(save_dir+img_name, res)
	return res




