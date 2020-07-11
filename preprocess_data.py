# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Script for data pre-processing."""

import tensorflow as tf 
import numpy as np
import cv2
from PIL import Image
import os
from scipy.io import loadmat,savemat
from renderer import face_decoder
from training.networks_recon import R_Net
from preprocess.preprocess_utils import *
from keras.utils import get_file
import bz2
from argparse import ArgumentParser
from training.training_utils import process_reals
import PIL

# Pretrained face reconstruction model from Deng et al. 19,
# https://github.com/microsoft/Deep3DFaceReconstruction
model_continue_path = 'training/pretrained_weights/recon_net'
R_net_weights = os.path.join(model_continue_path,'FaceReconModel')
config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'

def parse_args():
    desc = "Data Preprocess of DisentangledFaceGAN"
    parser = ArgumentParser(description=desc)

    parser.add_argument('--image_path', type=str, help='Training image path.')
    parser.add_argument('--save_path', type=str, default='./data' ,help='Save path for aligned images and extracted coefficients.')

    return parser.parse_args()


def main():
	args = parse_args()
	image_path = args.image_path
	# lm_path = os.path.join(args.image_path,'lm5p') # detected landmarks for training images should be saved in <image_path>/lm5p subfolder
	
	# create save path for aligned images and extracted coefficients
	save_path = args.save_path	
	if not os.path.exists(os.path.join(save_path,'img')):
		os.makedirs(os.path.join(save_path,'img'))
	if not os.path.exists(os.path.join(save_path,'coeff')):
		os.makedirs(os.path.join(save_path,'coeff'))

	# Load BFM09 face model
	if not os.path.isfile('./renderer/BFM face model/BFM_model_front_gan.mat'):
		transferBFM09()
	
	# Load standard landmarks for alignment
	lm3D = load_lm3d()


	# Build reconstruction model
	with tf.Graph().as_default() as graph:

		images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
		Face3D = face_decoder.Face3D() # analytic 3D face formation process
		coeff = R_Net(images,is_training=False) # 3D face reconstruction network

		with tf.Session(config = config) as sess:

			var_list = tf.trainable_variables()
			g_list = tf.global_variables()

			# Add batch normalization params into trainable variables 
			bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
			bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
			var_list +=bn_moving_vars

			# Create saver to save and restore weights
			resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
			res_fc = [v for v in var_list if 'fc-id' in v.name or 'fc-ex' in v.name or 'fc-tex' in v.name or 'fc-angles' in v.name or 'fc-gamma' in v.name or 'fc-XY' in v.name or 'fc-Z' in v.name or 'fc-f' in v.name]
			resnet_vars += res_fc

			saver = tf.train.Saver(var_list = var_list,max_to_keep = 100)
			saver.restore(sess,R_net_weights)

			for file in os.listdir(os.path.join(image_path)):
				if file.endswith('png'):
					print(file)

					full_img_path = os.path.join(image_path,file)

					# load images
					image = Image.open(full_img_path)

					# load landmarks
					landmarks = get_landmarks_dlib(full_img_path)

					face = 0

					for lm in landmarks:
						face = face + 1
						lm = np.reshape(lm,[5,2])
						print(lm)

						# align image for 3d face reconstruction
						align_img,_,_ = Preprocess(image,lm,lm3D) # 512*512*3 RGB image
						align_img = np.array(align_img)

						align_img_ = align_img[:,:,::-1] #RGBtoBGR
						align_img_ = cv2.resize(align_img_,(224,224)) # input image to reconstruction network should be 224*224
						align_img_ = np.expand_dims(align_img_,0)
						coef = sess.run(coeff,feed_dict = {images: align_img_})
						render_img,render_mask,render_landmark,_ = Face3D.Reconstruction_Block(coeff,256,1,progressive=False)

						mask = sess.run(render_mask,feed_dict = {images: align_img_})
						lma = sess.run(render_landmark,feed_dict = {images: align_img_})

						mask_img = os.path.join(save_path,'img',f'mask_{fname}')
						imask = (255*mask).astype('uint8')
						imask = PIL.Image.fromarray(imask, 'L')
						print("Saving mask " + mask_img)
						imask.save(mask_img, 'PNG')

						# align image for GAN training
						# eliminate translation and rescale face size to proper scale
						rescale_img = crop_n_rescale_face_region(align_img,coef) # 256*256*3 RGB image
						coef = np.squeeze(coef,0)

						# make save name corresponding to face
						fname = f'{face}_{file}'

						# save aligned images and extracted coefficients
						cv2.imwrite(os.path.join(save_path,'img',fname),rescale_img[:,:,::-1])
						#cv2.imwrite(os.path.join(save_path,'img',f'0_{fname}'),render_img)
						np.save(os.path.join(save_path,'img',f'mask_{fname}.npy'),mask)
						np.save(os.path.join(save_path,'img',f'lma_{fname}.npy'),lma)
						#cv2.imwrite(os.path.join(save_path,'img',f'2_{fname}'),render_landmark)
						savemat(os.path.join(save_path,'coeff',fname.replace('.png','.mat')),{'coeff':coef})


if __name__ == '__main__':
	main()