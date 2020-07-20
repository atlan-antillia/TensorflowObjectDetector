# CocoModelDownloader.py
#
#  Copyright (c) 2020 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# This is based on Tensorflow Object Detection API
# https://github.com/tensorflow/models
#    research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import zipfile

#from distutils.version import StrictVersion
#from collections import defaultdict
#from io import StringIO

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

#from object_detection.utils import ops as utils_ops

#from utils import label_map_util
#from utils import visualization_utils as vis_util

class CocoModelDownloader:

  def __init__(self):
    #self.MODEL_NAME   = 'ssd_mobilenet_v1_coco_2017_11_17'
    self.MODEL_NAME    = 'faster_rcnn_inception_v2_coco_2018_01_28'
    self.MODEL_FILE    = self.MODEL_NAME + '.tar.gz'
    self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


  def download(self):    
    # Download Model
    if not os.path.exists(self.MODEL_FILE):
      print("CocoModelDownloader.download start")
      print("Downloading a model file {}".format(self.MODEL_FILE))
      opener = urllib.request.URLopener()
      opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
      tar_file = tarfile.open(self.MODEL_FILE)

      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
          tar_file.extract(file, os.getcwd())
    else:
      print("Found a model file {}".format(self.MODEL_FILE))
    print("CocoModelDownloader.download end")
          
    
  def get_frozen_graph_path(self):
    path = self.MODEL_NAME + '/frozen_inference_graph.pb'
    return path


  def get_label_path(self):
    path= os.path.join('data', 'mscoco_label_map.pbtxt')
    return path
    

