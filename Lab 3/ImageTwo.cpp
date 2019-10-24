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

int main(int argc, char* argv[]){

  //load image into grey
  Mat image = imread("car2.png", 1);
  Mat grey_img;
  cvtColor( image, grey_img, CV_BGR2GRAY );
  Mat new_img;
  grey_img.copyTo(new_img);

  int size = atoi(argv[1]); // size of sq around pixel to det the neighbour
  for (int y = 0; y < grey_img.cols; y++){
    for (int x = 0; x < grey_img.rows; x++){

      vector<unsigned char> neighbours;

      for (int i = -size; i <= size; i++){
        for (int j = -size; j <= size; j++){
          if (!(y+i < 0 || y+i > grey_img.cols|| x+j < 0 || x+j > grey_img.rows)){
            neighbours.push_back(grey_img.at<uchar>(x+j, y+i));
          }

        }
      }
      int n = neighbours.size() / 2;
      nth_element(neighbours.begin(), neighbours.begin() + n, neighbours.end());
      //printf("%d/%d ,%d/%d\n",y,grey_img.cols, x, grey_img.rows - 1);
      new_img.at<uchar>(x,y) = neighbours[n];
    }
  }
  printf("made it!\n");
  imwrite("Fixed_car2.png", new_img);

  //wait for a key press until returning from the program
  //waitKey(0);

  //free memory occupied by image
  image.release();
  new_img.release();
  grey_img.release();

  return 0;
}
