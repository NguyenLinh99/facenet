from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
from mtcnn import MTCNN
from PIL import Image
import cv2
import glob
from scipy.spatial import distance_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
from sklearn import neighbors, datasets

def load_and_align_data(image_paths, pnet, rnet, onet, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    labels = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        label = os.path.split(os.path.dirname(image))[-1]
        labels.append(label)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images,labels

def aln_image(m_detector, paths, margin, image_size):
	results = []
	for path in paths:
		# print(path)
		img = misc.imread(os.path.expanduser(path), mode='RGB')
		# print(m_detector.detect_faces(img))
		img_size = np.asarray(img.shape)[0:2]
		faced = m_detector.detect_faces(img)[0]
		x, y, width, height = np.array(faced["box"])
		width += x
		height += y

		bb = np.zeros(4, dtype=np.int32)
		bb[0] = np.maximum(x-margin/2, 0)
		bb[1] = np.maximum(y-margin/2, 0)
		bb[2] = np.minimum(width+margin/2, img_size[1])
		bb[3] = np.minimum(height+margin/2, img_size[0])
		cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
		print(cropped.shape)
		aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
		prewhitened = facenet.prewhiten(aligned)
		results.append(prewhitened)
	return results

def load_dataset(folders):
	embs = np.ones((0,512))
	test = []
	listimage = []
	# folders = "/home/v000354/Documents/Linh/Face_recognition/facenet/data/images/train"
	for path, subdirs, files in os.walk(folders):
		for name in files:
			listimage.append(os.path.join(path, name))
		# print(listimage)
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
			facenet.load_model("/home/v000354/Documents/Linh/facenet/20180402-114759/20180402-114759.pb")
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			labels = []
			for idx in range(0, len(listimage), 10):	
				startidx = idx
				endidx = min(len(listimage),idx + 10)
				paths = listimage[startidx:endidx]
				# print(paths)
				# label = os.path.basename(os.path.dirname(paths[0]))
				images,tmp_labels = load_and_align_data(paths, pnet, rnet, onet,160,44)

				feed_dict = { images_placeholder: images, phase_train_placeholder:False }
				emb = sess.run(embeddings, feed_dict=feed_dict)
				embs = np.concatenate((embs,emb))
				labels += tmp_labels
			# print(embs)
			# np.save("faced.npy", [embs, labels])
			return embs, labels

# trainX, trainY = load_dataset("/home/v000354/Documents/Linh/facenet/data/images/aln_train")
# testX, testY = load_dataset("/home/v000354/Documents/Linh/facenet/data/images/aln_val")

# with open("train.dat", "wb") as f:
#     pickle.dump((trainX, trainY), f)
# with open("test.dat", "wb") as f:
#     pickle.dump((testX, testY), f)
# exit()

with open("train.dat", "rb") as f:
    trainX, trainY =  pickle.load(f)
with open("test.dat", "rb") as f:
    testX, testY =  pickle.load(f)

# trainY = np.array(trainY)
# uni = np.unique(trainY)[1]
# idxs = np.where(trainY == str(uni))

# uni_1 = np.unique(trainY)[1]
# idxs_1 = np.where(trainY == str(uni_1))
# print(distance_matrix(trainX[idxs], trainX[idxs_1]))
# np.save("test.npy", [testX, testY])
# print(np.array(trainX).shape)
# print(trainX.shape, len(trainY))
# exit()
# testX, testY = np.load("test.npy")

model = SVC(kernel='linear', probability=True)
print(np.array(trainX).shape)
model.fit(np.array(trainX), trainY)
yhat_train = model.predict(np.array(trainX))
yhat_test = model.predict(np.array(testX))
print(yhat_train[:10])
print(trainY[:10])
# score
score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
# clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# clf.fit(trainX, trainY)
# y_pred = clf.predict(testX)
# print("Predicted labels: ", y_pred[:10])
# print("Ground truth    : ", testY[:10])

# idx_min = np.argmin(distance_matrix(trainX, testX[:1]))
# y_pred = trainY[idx_min]
# y_true = testY[:1]
# print(y_pred, y_true)
# path = "/home/v000354/Documents/Linh/Face_recognition/facenet/data/images/aln_train/madonna/httpmediavoguecomrwblondesdarkbrowsmadonnajpg.png"
# path_1 = "/home/v000354/Documents/Linh/Face_recognition/facenet/data/images/aln_val/madonna/httpcdncdnjustjaredcomwpcontentuploadsheadlinesmadonnatalksparisattackstearsjpg.png"
# faced = []
# with tf.Graph().as_default():
# 	with tf.Session() as sess:
# 		facenet.load_model("/home/v000354/Documents/Linh/Face_recognition/facenet/20180408-102900/20180408-102900.pb")
# 		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
# 		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
# 		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
# 		image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# 		# image = cv2.resize(image, (160, 160))
# 		label = os.path.split(os.path.dirname(path))[-1]
# 		image_1 = cv2.cvtColor(cv2.imread(path_1), cv2.COLOR_BGR2RGB)
# 		# image_1 = cv2.resize(image_1, (160, 160))
# 		label_1 = os.path.split(os.path.dirname(path_1))[-1]
# 		faced.append(image)
# 		faced.append(image_1)
# 		feed_dict = { images_placeholder: np.array(faced), phase_train_placeholder:False }
# 		emb = sess.run(embeddings, feed_dict=feed_dict)
# 		print(distance_matrix(emb, emb))
# image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (160, 160))
# print(image.shape)
