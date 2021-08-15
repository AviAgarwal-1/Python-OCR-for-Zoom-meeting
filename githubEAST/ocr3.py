#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:21:29 2020

@author: tekhawk
"""
import pytesseract
import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0,'/home/tekhawk/githubEAST/')


import locality_aware_nms as nms_locality
import lanms


#%%
"""
import argparse


# Initializing Parser
parser = argparse.ArgumentParser(description ='sort some integers.')

# Adding Argument
parser.add_argument("-t","--test_data_path" ,type= str ,
                    default = '/home/tekhawk/githubEAST/demo_images/' ,
                    help ="path for test data")

parser.add_argument("-g", "--gpu_list" ,type = str ,
                    default = '1',
                    help ='gpu(s) to use')
parser.add_argument("-c", "--checkpoint_path" , type =str ,
                    default ='/home/tekhawk/githubEAST/checkpoint/resnet_v1_50.ckpt',
                    help = "path for trained model checkpoint")
parser.add_argument("-o", "--output_dir" ,type =str,
                    default ='/home/tekhawk/githubEAST/output/',
                    help= "path to output directory ")

parser.add_argument("-w","--no write images", type = bool ,
                    default = False,
                    help = "write over image , default = false")

parser.add_argument("-r","--tes_loc" , type=str,
                    default = '' ,
                    help = "tesseract location directory")

args = parser.parse_args()


"""
#%%

####Delete all flags before declare#####
"""
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.compat.v1.app.flags.FLAGS)
"""

tf.compat.v1.app.flags.DEFINE_string('test_data_path','/home/tekhawk/githubEAST/demo_images','')
tf.compat.v1.app.flags.DEFINE_string('gpu_list', '0','')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_path', '/home/tekhawk/githubEAST/checkpoint/east_icdar2015_resnet_v1_50_rbox','')
tf.compat.v1.app.flags.DEFINE_string('output_dir', '/home/tekhawk/githubEAST/output','')
tf.compat.v1.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
#tf.compat.v1.app.flags.DEFINE_string("tes_loc", '')

#FLAGS = tf.compat.v1.app.flags.FLAGS

#%%

import model
from icdar import restore_rectangle

FLAGS = tf.compat.v1.app.flags.FLAGS

#%%




#pytesseract.pytesseract.tesseract_cmd

#%%

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
#    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

#max_slide_len = 2400
def resize_image(im, max_side_len=24000):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

#%%
def main(argv = None):
    global boxes
    import os
#    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
#    os.environ['tf_CPP_MIN_LOG_LEVEL'] = '2'

    try:
#        os.makedirs(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.compat.v1.get_default_graph().as_default():
        input_images = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, None, None, 3], name='input_images' )
        global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.compat.v1.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.compat.v1.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.compat.v1.train.Saver(variable_averages.variables_to_restore())

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
#            ckpt_state = tf.compat.v1.train.get_checkpoint_state(FLAGS.checkpoint_path)
            ckpt_state = tf.compat.v1.train.get_checkpoint_state(FLAGS.checkpoint_path)
#           model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                reader2(boxes, im)
                


                # save to file
                if boxes is not None:
                    res_file = os.path.join(
#                        FLAGS.output_dir,
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
#                if not FLAGS.no_write_images:
                if not FLAGS.no_write_images:
#                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])


#%% reader1.0
def reader( orig, display = True):
    global boxes
    roi =[]
    results ,ext_txt = [] , []
        # scale the bounding box coordinates based on the respective
        # ratios
#        startX = int(startX * rW)
#        startY = int(startY * rH)
#        endX = int(endX * rW)
#        endY = int(endY * rH)


        # apply a bit of padding surrounding the bounding box
        # computing the deltas in both the x and y directions
#        dX = int((endX - startX) * args["padding"])
#        dY = int((endY - startY) * args["padding"])

        # apply padding to each side of the bounding box
#        startX = max(0, startX - dX)
#        startY = max(0, startY - dY)
#        endX = min(origW, endX + (dX * 2))
#        endY = min(origH, endY + (dY * 2))
    
            # extract the actual padded ROI
#            roi = orig[startY:endY, startX:endX]
  
        # configuration for tesseract
        # extracting text from image with individual boxes as strings
#    pytesseract.pytesseract.tesseract_cmd = r'/usr/share/tesseract-ocr/4.00/tessdata'
    config = ("--oem 1 ")
    text = pytesseract.image_to_string(roi,)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
    results.append(text)

        # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])
#    results = word_arr(results)


    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("{}\n".format(text))
        ext_txt.append(text)
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image

        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # show the output image
        if (display == True):
            cv2.imshow("Text Detection", output)

            cv2.waitKey(0)


    return ext_txt



#%% reader2.0
    
def reader2(boxes, orig, display=True):
    text =[]
    

    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
            continue
    
#        roi1 = _roi2(orig,boxes)
        roi1 = roi(orig,box)
        roi1 = preprocess2(roi1)
#        pytesseract.pytesseract.tesseract_cmd = r'/usr/share/tesseract-ocr/4.00/tessdata'
        config = ("--oem 1  --psm 7")
        text = pytesseract.image_to_string(roi1, config=config)
        print("THIS IS THE TEXT:"+ text)
        
        cv2.imshow("text", roi1)
        cv2.waitKey(300)
       
    return 0

#%%

def _box_validator(boxes):
    valid_box = []
    
    for box in boxes:
        box = sort_poly(box.astype(np.int32))
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
            continue
        valid_box.append(box)
        
    return valid_box
#%% roi1.0

def roi(img,box):
    global _debug2
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    points = box.astype(np.int32).reshape((-1, 1, 2))
    points2 = expand2(points, 0.45)
    
# best results: points2(0.4) preprocess 2

    #method 1 smooth region
    cv2.drawContours(mask, [points2], -1, (255, 255, 255), -1, cv2.LINE_AA)
 
    #method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))
 
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points2) # returns (x,y,w,h) of the rect
#    rect2 = expand(rect, 5) 
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
#   cropped = res[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]   
    
    if(rect[3] > 40 or rect[3] < 30):
        cropped,new_w,new_h= _resize(cropped,rect[3])

        
        
    ## crate the white background of the same size of original image
#    cv2.bitwise_not(wbg,wbg, mask=mask)
    # overlap the resulted cropped image on the white background
#    dst = wbg+res
    _debug2.append(points)
#    cv2.imshow("Cropped", cropped )
#    cv2.waitKey(500)
    
    return cropped

#%% margin
    
def expand(points, margin):
    # supose points is 1, p2, p3, p4
    _point = []
    _point.append(points[0] - margin) 
    _point.append(points[1] - margin)
    _point.append(points[2] + margin*2)
    _point.append(points[3] + margin*2)
    
    return _point 

def expand2(points, margin):
    # supose points is 1, p2, p3, p4
    points[0][0][0] = points[0][0][0] - margin 
    points[0][0][1] = points[0][0][1] - margin
    points[1][0][0] = points[1][0][0] + margin*2.3
    points[1][0][1] = points[1][0][1] - margin*2.3
    points[2][0][0] = points[2][0][0] + margin*2.3
    points[2][0][1] = points[2][0][1] + margin*4.5
    points[3][0][0] = points[3][0][0] - margin 
    points[3][0][1] = points[3][0][1] + margin*4.5
    return points

#%% roi2.0 dependent on roi1.0

def _roi2(img,boxes):
    
    global _debug
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_bg = 255*np.ones_like(img)
    
    
    
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.medianBlur(thresh, 1)
    kernel = np.ones((10, 20), np.uint8)
    img_dilation = cv2.dilate(blur, kernel, iterations=1)
#    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = _box_validator(boxes)
 
    for box in boxes:
        
            points = box.astype(np.int32).reshape((-1, 1, 2))
            rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
            _roi = roi(img,box)
            if (rect[3] < 23): 
                _roi ,new_w,new_h= _resize(_roi,rect[3])
                white_bg[rect[1] : rect[1] + new_h , rect[0] : rect[0] + new_w] = _roi 
            else:    
            #--- paste ROIs on image with white background 
                white_bg[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = _roi       # white_bg[y:y+h, x:x+w] = roi
            _debug.append(rect)
    cv2.imshow('white_bg_new', white_bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return white_bg

#%% preprocess roi
    
def preprocess(img):
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)   
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)   
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

def preprocess2(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
#%% resize

def _resize(img,h):
    _percent = 31/h
    scale_percent = _percent * 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim,interpolation = cv2.INTER_CUBIC) 
    return (resized , width, height)
#%%
if __name__ == '__main__':
    _debug , _debug2 = [], []
    tf.compat.v1.app.run()
    
