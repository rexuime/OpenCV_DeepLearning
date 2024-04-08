-----------------------------------------------------------------------------------------------------------------------------------------------------------------
Deep Learning Object Detection
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'General_Object_Detection':

A general example of object detection. It isn't the best due to the limited pool of training data MobileNet SSD has but it is a simple example for beginners.
Run the commands below to execute the Python source code in this folder.

python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.proto.txt --model MobileNetSSD_deploy.caffemodel --image dogsandcats.jpg
python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.proto.txt --model MobileNetSSD_deploy.caffemodel --image dogsandcats2.jpg

If you have an image you want to test, just replace "dogsandcats.jpg" or "dogsandcats2.jpg" with the path to your image.

There are also examples of the resulting images from executing the code in this folder as well.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'Circle_Detection':

A more specific example of object detection. Code for detecting simple circles in images.
Run the commands below to execute the Python source code in this folder.

python circle_detect.py shapes.jpg
python circle_detect.py shapes2.jpg

If you have an image you want to test, just replace "shapes.jpg" or "shapes2.jpg" with the path to your image.

There are also examples of the resulting images from executing the code in this folder as well.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'Deep_Learning_Training'

A general example of training a deep learning network. Is a very simple example and doesn't show much. Includes an example image of the digits used from MNIST
as well as an example image of the training results.

There are two different Python files that do basically the same thing.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
