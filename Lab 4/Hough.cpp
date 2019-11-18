#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc */

using namespace cv;
using namespace std;

// This will be the hand written function for convolution of an image
Mat convolution (Mat base_img, Mat kernel);

int main(){

  // read img
  Mat image = imread("coins1.png", 0);

  // set up tranform kernel
  Mat yKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat xKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);

  Mat ys = convolution(image, xKernel);
  Mat xs = convolution(image, yKernel);

  xs.convertTo(xs,CV_8UC1);
  ys.convertTo(ys,CV_8UC1);

  Mat combo = xs + ys;
  //construct a window for image display
  namedWindow("Display window", WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", xs);

  //wait for a key press until returning from the program
  waitKey(0);

  imshow("Display window", ys);
  waitKey(0);

  imshow("Display window", combo);
  waitKey(0);

  //free memory occupied by image
  image.release();
  xKernel.release();
  yKernel.release();
  xs.release();
  ys.release();
  return 0;
}

Mat convolution(Mat base_img, Mat kernel){

  Mat new_img = base_img.clone(); //create output image
  new_img.convertTo(new_img, CV_32F); //convert to float so we can get larger numbers

  //Loop through the image
  for (int y = 0; y < base_img.rows; y++){
    for (int x = 0; x < base_img.cols; x++){

      //reset sum at the start of a new pixel
      float sum = 0;

      //Loop through the kernel
      for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
          //check if you go out of image range
          if (!(y+i < 0 || y+i > base_img.rows || x+j < 0 || x+j > base_img.cols)){
            sum += (base_img.at<uchar>(y+i,x+j) * kernel.at<float>(i+1,j+1));
          }
        }
      }
      /*
      if (sum > 255){
        sum = 255;
      } else if (sum < 0){
        sum = 0;
      }
      */

      //set the sum to the new image
      new_img.at<float>(y,x) = sum;
    }
  }

  //Mat norm_img;
  //Norm image
  //normalize(new_img, norm_img, 0, 255, NORM_MINMAX);

  return new_img;
}
