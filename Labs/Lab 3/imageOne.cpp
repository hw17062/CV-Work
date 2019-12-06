#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include "filter2d.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc */

using namespace cv;
using namespace std;

int main(){

  //load image into grey
  Mat image = imread("car1.png", 1);
  Mat grey_img;
  Mat dest;
  Mat saturated;
  Mat mask;
  cvtColor( image, grey_img, CV_BGR2GRAY );

  //convert to float of convolve
  //image.convertTo(image, CV_32F);

  // create kernel for convolve
  //Mat kernel = Mat(3 ,3, CV_32F ,Scalar(1,1,1));
  //kernel = kernel / 9;
  //cout << "kernel = " << endl << " " << kernel << endl << endl;

  //get blurred image
  Mat blurred = blur(image);

  multiply(grey_img, 3, saturated);

  //mask = grey_img- blurred;

  dest = grey_img + grey_img - blurred;

  //imwrite("mask.jpg", mask);
  imwrite("blurred.jpg", blurred);
  imwrite("img1_fixed.jpg", dest);

  //wait for a key press until returning from the program
  //waitKey(0);

  //free memory occupied by image
  image.release();
  //kernel.release();
  blurred.release();
  return 0;
}
