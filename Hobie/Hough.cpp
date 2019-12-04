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
Mat houghCircles(Mat grad, Mat dir, Mat img);
Mat direction (Mat xs, Mat ys);
int ***malloc3dArray(int dim1, int dim2, int dim3);
Mat printPixel(Mat image);

int main(){

  // read img
  Mat img = imread("dart9.jpg", 1);
  Mat image,img2;
  img2 = img.clone();
  cvtColor( img, img, CV_BGR2GRAY );
  GaussianBlur(img, image, Size(7 ,7 ),0 ,0 );
  // set up tranform kernel
  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);
  printf("finished conv\n");
  Mat grad;
  grad = xs.mul(xs) + ys.mul(ys);
  sqrt(grad, grad);
  Mat norm = grad.clone();
  //normalize(grad, grad, 0, 255, NORM_MINMAX);

  //cout << "Before Normalization" << endl;
  //printPixel(norm);
  cout << "After Normalization" << endl;
  //printPixel(grad);

  Mat angle = direction(xs, ys);
  Mat canny ;
  //Canny(img, canny, 110, 380, 3);
  Mat preH;// preH = printPixel(grad);
  // Now start working on Hough transformation
  Mat hough = houghCircles(grad, angle, img2);

  xs.convertTo(xs,CV_8UC1);
  ys.convertTo(ys,CV_8UC1);
  norm.convertTo(norm,CV_8UC1);
  grad.convertTo(grad,CV_8UC1);
  angle.convertTo(angle,CV_8UC1);
  canny.convertTo(canny,CV_8UC1);
  hough.convertTo(hough,CV_8UC1);

  //imshow("Gradient Magnitude", grad);
  imshow("hough", hough);
  waitKey(0);

  //imwrite("Before Normalization.jpg", norm);
  imwrite("Gradient Magnitude.jpg", grad);
  //imwrite("GBlur.jpg", image);
  imwrite("Degrees.jpg", angle);
  //imwrite("Canny.jpg", canny);
  imwrite("Hough.jpg", hough);

  //free memory occupied by image
  image.release();xKernel.release();yKernel.release();xs.release();ys.release();
  grad.release();angle.release();canny.release();hough.release();preH.release();img2.release();
  return 0;
}


Mat houghCircles(Mat grad, Mat dir, Mat img){

  Mat img_h = img.clone();
  int a, b,c,d;
  int voteTh = 16,pixTh = 60 ;
  int rmin = 100, rmax = 130;
  int ***acc;
  acc = malloc3dArray(grad.rows, grad.cols, rmax);
  int acc2D [grad.rows][grad.cols];
  float pix ;

  //INTILIALISING ACC
  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      for(int r = 0 ; r < rmax ; r++)
      {
        acc[x][y][r] = 0;
      }
    }
  }

  //VOTING
  for(int x = 210 ; x < grad.rows-80 ; x++)
  {
    for(int y = 10 ; y < grad.cols-220 ; y++)
    {
      pix = grad.at<float>(x,y);
      if (pix >pixTh)
      {
        float direction = dir.at<float>(x,y);
        for(int r = rmin ; r < rmax ; r++)
        {
          for(float t = direction - 0.1 ; t < direction + 0.1; t += 0.02)
          {
            a = (int)(x - r * cos(t));
            b = (int)(y - r * sin(t));

            c = (int)(x + r * cos(t));
            d = (int)(y + r * sin(t));
            //VOTING -------
            if(a>0 && a<grad.rows && b>0 && b<grad.cols){  acc[a][b][r] += 1;}
            if(c>0 && c<grad.rows && d>0 && d<grad.cols){  acc[c][d][r] += 1;}
          }
        }
      }
    }
  }

  // CONERT HOUGH-3D to HOUGH-2D
  int vote = 0, rad = 0;
  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      vote = 0; rad = 0;acc2D[x][y]=0;
      for(int r = rmin ; r < rmax ; r++)
      {
        if(acc[x][y][r] > vote)
        {
          vote = acc[x][y][r];
          acc2D[x][y] = r;
        }
      } rad = acc2D[x][y];
      //if(rad > 0){  cout << x<<" "<<y<<" "<<" "<<acc2D[x][y]<<" "<<acc[x][y][rad] << endl;}

    }
  }
  cout << endl;
  // PRINTING CIRCLE CENTRES > THRESHOLD VALUE
  vote=0;
  for(int x = 0 ; x < grad.rows ; x++)
  {
    for(int y = 0 ; y < grad.cols ; y++)
    {
      //for(int r = rmin ; r < rmax ; r++)
      //{
        rad = acc2D[x][y];
        vote = acc[x][y][ rad ];
        if(vote > voteTh){
          circle(img_h, Point(x,y), rad, cvScalar(255,0,0), 2);
          //cout << x<<" "<<y<<" "<<" "<<r<<" "<<acc[x][y][r] << endl;
        }
      //}
    }
  }
  return img_h;

}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}

    }
    return array;
}

Mat direction (Mat xs, Mat ys){

  Mat theta = xs.clone();
  float dx,dy, ang;

  for (int x = 0; x < xs.rows; x++){
    for (int y = 0; y < xs.cols; y++){
      dx = xs.at<float>(x,y);
      dy = ys.at<float>(x,y);
      //ang = atan(dy/dx)* 180 / M_PI;
      theta.at<float>(x,y) = fastAtan2(-dy,dx);//fastAtan2(dy,dx);// in radians
    }
  }
  return theta;

}

Mat printPixel(Mat image){
  Mat copy = image.clone();
  Mat copy2 = Mat(image.size(), CV_8UC1);

  float pix = 0.0f;
  for(int x =  0; x < copy.rows ; x++)
  {
    cout << x << "}  ";
    for(int y =  0; y < copy.cols ; y++)
    {
      pix = copy.at<float>(x,y);
      //pix = (pix > 50)?255:0;
      //if(x==0 || y==0 || x>copy.rows-2 || y>copy.cols-2)  pix=0;
      cout << pix <<" : ";
      //copy2.at<uchar>(x,y) = pix;

    }cout << endl<<endl;
  }cout << endl<< endl;

  //imshow("copy2", copy2);
  //imwrite("copy2.jpg",copy2);
  return copy2;

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
