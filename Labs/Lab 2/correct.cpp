/////////////////////////////////////////////////////////////////////////////
//
// correct.cpp
// TOPIC: load Mandill images that are corruped then correct them
//
// Getting-Started-File for OpenCV
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

Mat correctionOne(Mat imageCorrupted);
Mat correctionTwo();


//The main will take the arg of what image to correct between 0-3
int main(int argc, char* argv[]) {
  // Check usage
  if (argc != 2) {
    fprintf(stderr, "Usage: %s file_Number\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  Mat imageCorrupted;
  Mat imageCorrected;
  int whichImg = atoi(argv[1]);

  if (whichImg == 0) {

    Mat imageCorrupted = imread("mandrill0.jpg", 1);
    imageCorrected = correctionOne(imageCorrupted);
  } else if(whichImg == 1){
    Mat imageCorrupted = imread("mandrill1.jpg", 1);
    imageCorrected = correctionTwo();
  }

  //construct a window for image display
  namedWindow("Display window", WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", imageCorrected);

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image
  imageCorrupted.release();
  imageCorrected.release();

  return 0;
}

Mat correctionOne(Mat imageCorrupted){
  // Read image from file
  //Mat imageNormal = imread("mandrill.jpg", 1);

  for(int y=0; y<imageCorrupted.rows; y++) {
   for(int x=0; x<imageCorrupted.cols; x++) {
     uchar pixelBlue = imageCorrupted.at<Vec3b>(y,x)[0];
     uchar pixelGreen = imageCorrupted.at<Vec3b>(y,x)[1];
     uchar pixelRed = imageCorrupted.at<Vec3b>(y,x)[2];

     imageCorrupted.at<Vec3b>(y,x)[0] = pixelRed;
     imageCorrupted.at<Vec3b>(y,x)[1] = pixelBlue;
     imageCorrupted.at<Vec3b>(y,x)[2] = pixelGreen;
/*
     if (x == 236 && y == 256){
       printf("The value on the normal image is: (%d,%d,%d).\n The Value on the corrupted image is (%d,%d,%d)\n",
        imageNormal.at<Vec3b>(y,x)[0],imageNormal.at<Vec3b>(y,x)[1], imageNormal.at<Vec3b>(y,x)[2],
        imageCorrupted.at<Vec3b>(y,x)[0],imageCorrupted.at<Vec3b>(y,x)[1], imageCorrupted.at<Vec3b>(y,x)[2]);
     }
     */
    }
  }

  return imageCorrupted;
}


Mat correctionTwo(){
  Mat imageCorrupted = imread("mandrill0.jpg", 1);
  //Mat imageNormal = imread("mandrill.jpg", 1);

  for(int y=0; y<imageCorrupted.rows; y++) {
   for(int x=0; x<imageCorrupted.cols; x++) {
     uchar pixelRed = imageCorrupted.at<Vec3b>(y,x)[2];

       if (x > 20 && y > 20){
         uchar pixelRed = imageCorrupted.at<Vec3b>(y-20,x-20)[2];
         imageCorrupted.at<Vec3b>(y,x)[2] = pixelRed;
       }
     }
   }
   imageCorrupted = correctionOne(imageCorrupted);
   for(int y=0; y<imageCorrupted.rows; y++) {
    for(int x=0; x<imageCorrupted.cols; x++) {
      uchar pixelBlue = imageCorrupted.at<Vec3b>(y,x)[0];
      uchar pixelGreen = imageCorrupted.at<Vec3b>(y,x)[1];
      uchar pixelRed = imageCorrupted.at<Vec3b>(y,x)[2];

      imageCorrupted.at<Vec3b>(y,x)[0] = pixelRed -20;
      imageCorrupted.at<Vec3b>(y,x)[1] = pixelBlue;
      imageCorrupted.at<Vec3b>(y,x)[2] = pixelGreen - 20;

      }
    }
   return imageCorrupted;
}
