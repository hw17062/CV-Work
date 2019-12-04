#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc, atan */

#define PI 3.14159265

using namespace cv;
using namespace std;



// This will be the hand written function for convolution of an image
Mat convolution (Mat base_img, Mat kernel);
void solbet();
void hough(Mat xs, Mat ys, Mat mag, Mat grad);
Mat mat2gray(const cv::Mat&);
void threshold(Mat& img);

int main(){
  solbet();
  return 0;
}

//normalisied image for grey image showing/saving
Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

//thresholds the image sent to it. if above = 255 else = 0
void threshold(Mat& img){
  img = mat2gray(img);
  float threshold = 65;
  for (int y = 0; y < img.rows; y++){
    for (int x = 0; x < img.cols; x++){
      if (img.at<uchar>(y,x) > threshold)  img.at<uchar>(y,x) = 255;
      else img.at<uchar>(y,x) = 0;
    }
  }
}


void solbet(){

  // read img
  Mat image = imread("coins1.png", 0);

  // set up tranform kernel
  Mat yKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat xKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);

  Mat ys = convolution(image, xKernel);
  Mat xs = convolution(image, yKernel);


  // now Calc the Magnitude
  Mat mag;
  mag = xs.mul(xs) + ys.mul(ys);

  sqrt(mag, mag);

  // Now calc the gradient
  Mat grad = Mat::zeros(xs.rows, xs.cols, CV_32F);

  //loop through performing atan() to get the direction
  for (int y = 0; y < grad.rows; y++){
    for (int x = 0; x < grad.cols; x++){
      grad.at<float>(y,x) = abs(atan2(ys.at<float>(y,x), xs.at<float>(y,x)));
      //grad.at<float>(y,x) = (grad.at<float>(y,x) *180 / PI);
    }
  }
  //Mat gradN;
  //gradN.convertTo(gradN,CV_8UC1);

  //construct a window for image display
  namedWindow("Display window", WINDOW_AUTOSIZE);

  //visualise the loaded image in the window

  // imshow("Display window", image);
  // waitKey(0);
  //
  // imshow("Display window", mat2gray(xs));
  // waitKey(0);
  //
  // imshow("Display window", mat2gray(ys));
  // waitKey(0);
  //
  // imshow("Display window", mat2gray(mag));
  // waitKey(0);
  //
  // imshow("Display window", mat2gray(grad));
  // waitKey(0);

  hough(xs, ys, mag, grad);

  //free memory occupied by image
  image.release();
  xKernel.release();
  yKernel.release();
  xs.release();
  ys.release();
  mag.release();
  grad.release();

}

void hough(Mat xs, Mat ys, Mat mag, Mat grad){
  threshold(mag);

  imshow("Display window", mat2gray(mag));
  waitKey(0);
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

      //set the sum to the new image
      new_img.at<float>(y,x) = sum;
    }
  }

  //Mat norm_img;
  //Norm image
  //normalize(new_img, new_img, 0, 255, NORM_MINMAX);

  return new_img;
}
