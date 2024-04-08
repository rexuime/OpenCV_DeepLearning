import sys
import cv2 as cv
import numpy as np
def main(argv):
 
   # Default image if argument isn't given
   default_file = 'shapes.jpg'
   # Check for given image
   filename = argv[0] if len(argv) > 0 else default_file
   # Loads an image
   src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
   # Check if image is loaded fine
   if src is None:
      print ('Error opening image!')
      print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
      return -1
   
   # Converts the input image from BGR color space to grayscale
   gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
   
   # Median blurring on the grayscale image using a kernel size of 5x5
   gray = cv.medianBlur(gray, 5)
   
   # Detect where circles are and store their centers and radii
   rows = gray.shape[0]
   circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
   param1=100, param2=30,
   minRadius=1, maxRadius=30)
   
   
   if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
         center = (i[0], i[1])
         # circle center
         cv.circle(src, center, 1, (0, 100, 100), 3)
         # circle outline
         radius = i[2]
         cv.circle(src, center, radius, (255, 0, 255), 3)
   
   
   cv.imshow("detected circles", src)
   cv.waitKey(0)
   
   return 0

if __name__ == "__main__":
   main(sys.argv[1:])