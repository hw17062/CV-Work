/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: basic thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;

using namespace cv;

int main(int argc, char *argv[]) {

  // Read image from file
  Mat image = imread("mandrill.jpg", 1);

  int threshLo = stoi(argv[1]);
  int threshHi = stoi(argv[2]);
  // Convert to grey scale
  Mat gray_image;
  cvtColor(image, gray_image, CV_BGR2GRAY);


  //int thresh = 100;
  // Threshold by looping through all pixels
  for (int y = 0; y<gray_image.rows; y++) {
    for (int x = 0; x<gray_image.cols; x++) {
      uchar pixel = gray_image.at<uchar>(y, x);
      if (pixel > threshHi ) gray_image.at<uchar>(y, x) = 255;
      else if ( pixel < threshHi && pixel > threshLo) gray_image.at<uchar>(y, x) = 125;
      else if (pixel < threshLo) gray_image.at<uchar>(y, x) = 0;
  } }

  char fileName[50];
  sprintf (fileName, "thr%d-%d.jpg", threshLo, threshHi);
  //Save thresholded image
  imwrite(fileName, gray_image);

  return 0;
}
