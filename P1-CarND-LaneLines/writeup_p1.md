# **Finding Lane Lines on the Road** 

## Writeup 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

First, I converted the images to grayscale:
[grayscale]: ./gray.png

Then I added a gaussian filter with a kernel of 3, as I figured out, that 3 brought the best result:
[gaussian]: ./gaussian.png

After that I used the Canny Edge Detection with a low threshold of 100 and a high treshold of 150. This was the best result over all images and videos:
[canny_edge]: ./canny_edge.png

Then I cut out my region of interest. I decided to start from to bottom left and right corner with an offset of 80 pixels. And in the middle of the picture (height/2) with an offset of 300 pixels from left and right. This step should be more dynamic in the future, to be capable of different lane width and different image scales. 
[region]: ./region_of_interest.png

The fifth step is the Hough Transformation.
[hough]: ./hough.png

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps:
* calculate the slope of each recognzied line
* seperate the line by positive or negative slope to left and right lane, but only if the slope is between +-0.4 and +-0.9, in order to filter line in the probably right direction
* calculate the mean of slope and all x and y coordinates of both sides to do some extrapolation
* only if left or right are lines identified (not nan), calculate the intercept and the lowest and highest x-point; y is static at the bottom of the image and for now 50 pixels below the center of the image -> needs to be more dynamic


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lines are not straight but in a curve.

Another shortcoming could be the smoothing of the lane position from one image to another.

Another shortcoming could be in case of no lane detection, that we estimate the position by taking the line from the previous image.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to smooth the line positions from one image to another, by defining a maximum difference with something like +-10% for the lowest and highest point of the line.

Another potential improvement could be to separate the drawn line to 2 or more segments, in order to stick closer to the line especially in curves.
