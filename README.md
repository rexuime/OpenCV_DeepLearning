Deep Learning Object Detection

Learning how to do this in OpenCV using MobileNet SSD.

In the "General_Object_Detection" folder, run the commands:

python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.proto.txt --model MobileNetSSD_deploy.caffemodel --image dogsandcats.jpg

python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.proto.txt --model MobileNetSSD_deploy.caffemodel --image dogsandcats2.jpg

To see a general example of object detection. It isn't the best due to the limited pool of training data MobileNet SSD has but it is a simple example for beginners.
