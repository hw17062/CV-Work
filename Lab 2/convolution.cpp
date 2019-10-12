#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc */

using namespace cv;
using namespace std;


int main(int argc, char* argv[]) {

  //get image and convert to floats so we can do math
  Mat image = imread("mandrill.jpg", 0);
  image.convertTo(image, CV_32F);

  //create kernel image
  Mat kernel(3,3, CV_32F, Scalar(1,1,1));
  kernel = kernel / 9;

  Mat section;

  //def edited image
  Mat editedimage = image.clone();
  //editedimage.convertTo(CV_8UC1)
  //section.convertTo(image, CV_32F);

  /*
  cout << "section = " << endl << " " << section << endl << endl;
  cout << "kernel = " << endl << " " << kernel << endl << endl;
  cout << "section.mul(kernel) = " << endl << " " << section.mul(kernel) << endl << endl;
  cout << "section.mul(kernel).sum() = " << endl << " " << sum(section.mul(kernel))[0] << endl << endl;
  */


  for (int row = 0; row <= image.rows - 1; row++){
    for (int col = 0; col <= image.cols- 1; col++){

      //rect (x, y, width, height)
      if ((row != 0 && col != 0) && (row != image.rows -1 && col != image.cols -1)) {
        section = image(Rect(row - 1, col - 1, 3, 3)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel))[0]);
      }
      else if (row == 0 && col == 0){ //top cornor case
        section = image(Rect(0, 0, 2, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(1, 1, 2, 2))))[0]);
      }
      else if (row == image.rows - 1&& col == image.cols - 1){ //bottom cornor case
        section = image(Rect(row - 1, col - 1, 2, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 0, 2, 2))))[0]);
      }
      else if (row == image.rows - 1 && col == 0){ //bottom cornor case
        section = image(Rect(row - 1, col, 2, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 0, 2, 2))))[0]);
      }
      else if (row == 0 && col == image.cols - 1){ //bottom cornor case
        section = image(Rect(row, col-1, 2, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 0, 2, 2))))[0]);
      }
      else if (row == 0){ //row edge top
        section = image(Rect(0, col, 3, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 1, 3, 2))))[0]);
      }
      else if (col == 0){ //col edge left
        section = image(Rect(row, 0, 2, 3)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(1, 0, 2, 3))))[0]);
      }
      else if (row == image.rows - 1) { //bottom row case
        section = image(Rect(row -1 , col-1, 2, 3)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 0, 2, 3))))[0]);
      }
      else if (col == image.cols - 1){ //bottom cornor case
        section = image(Rect(row-1, col-1, 3, 2)).clone();
        editedimage.at<float>(col,row) = round(sum(section.mul(kernel(Rect(0, 0, 3, 2))))[0]);
      }

    }
  }



  //construct a window for image display
  //namedWindow("Display window", WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  editedimage.convertTo(editedimage, CV_8UC1);
  //imshow("Display window", editedimage);

  imwrite("convolution_grey.jpg", editedimage);

  //wait for a key press until returning from the program
  //waitKey(0);

  //free memory occupied by image
  image.release();
  editedimage.release();
  kernel.release();
  section.release();

  return 0;
}
