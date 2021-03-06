<html>
<body>
<h1>TensorflowObjectDetector</h1>
<font size=3><b>
This is a simple python class TensorflowObjectDetector based on Tensorflow Object Detection API.<br>
</b></font>
<br>
<font size=2>
We have downloaded <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a>,  
You have to instal tensorflow>=1.15.<br>

<table style="border: 1px solid red;">
<tr><td>
<font size=2>
pip install tensorflow==1.15<br>
pip install Cython<br>
git clone https://github.com/tensorflow/models.git -b v1.13.0<br>

</font>
</td></tr>


</table>


Run TensorflowObjectDetector.py script to detect objects in an image in the following way.<br><br>
<b>
>python TensorflowObjectDetector.py .\images\img.png<br>
</b>
<br>
In this case, we use CocoModelDownloader class and download the followng file:
  'faster_rcnn_inception_v2_coco_2018_01_28.tar.gz'<br>
from 'http://download.tensorflow.org/models/object_detection/'.

<br>
<br>
If you would like to use your own 'frozen_inference_graph.pb', and 'label_map.pbtxt', please run
the script in the following way:<br>
<br>
<b>
>python TensorflowObjectDetector.py .\images\img.png output_dir frozen_inference_graph.pb label_map.pbtxt<br>
</b>
<br>

Example 1<br>
>python TensorflowObjectDetector.py .\images\im.png
<br><br>
<img src="./detected/img.png" width="80%">
<br>
 This detector seems to be missing a lot of small objects, and a large bounding box of label "person:55%" in center of the image above 
 is a false classification.<br>
 

This object detection image can be compared with the images by <a href="https://github.com/atlan-antillia/EfficientDetector/blob/master/output/img.png">EfficientDet:EfficientDetector</a>
 and <a href="https://github.com/atlan-antillia/DETR/blob/master/detected/img.png">DETR: DetectionTransformer</a>.<br>

Which object detector should we use to get a much better result?<br>
 
<br>
Example 2<br>
>python TensorflowObjectDetector.py .\images\ShinJuku.jpg
<br><br>
<img src="./output/ShinJuku.jpg" width="80%">
<br>
<br>
Example 3<br>

>python TensorflowObjectDetector.py .\images\ShinJuku2.jpg
<br><br>
<img src="./output/ShinJuku2.jpg" width="80%">
<br>
<br>
Example 4<br>

>python TensorflowObjectDetector.py .\images\Takashimaya2.jpg
<br><br>
<img src="./output/Takashimaya2.jpg" width="80%">
<br><br>
You can specify input_image_dir ouput_image_dir in the following way:<br><br>
<b>
>python TensorflowObjectDetector.py input_image_dirs ouput_image_dir
</b>

</body>
</html>