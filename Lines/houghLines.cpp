#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <math.h>       /* round, floor, ceil, trunc */


using namespace cv;
using namespace std;


int ***malloc3dArray(int dim1, int dim2, int dim3);

Mat threshold(Mat grad);
Mat gradient(Mat image);
Mat direction (Mat image);
Mat convolution (Mat base_img, Mat kernel);

Mat houghLines(Mat grad, Mat dir, Mat img);


//*************************************************************************************************************************************************************

int main(int argc, const char** argv){

  // read img
  //Mat img = imread("dart14.jpg", 1);

  Mat img = imread(argv[1], 1);
  Mat img2=img.clone();

  Mat gray;
  cvtColor( img, gray, CV_BGR2GRAY );
  //GaussianBlur(gray, gray, Size(9 ,9 ),0 ,0 );

  Mat grad = gradient(gray);      //pixel range 0-255
  Mat angle = direction(gray);    //in radians
  Mat thresh = threshold(grad);   //pixel either 0 or 255

  Mat hline = houghLines(thresh, angle, img2);

  //hline.convertTo(hline,CV_8UC1);
  //imshow("hough line", hline);
  //imwrite("Hough line.jpg", hline);
  waitKey(0);
  return 0;
}

//*************************************************************************************************************************************************************


//*************************************************************************************************************************************************************

Mat houghLines(Mat grad, Mat dir, Mat img)
{

  int voteTh = 2;


  Mat img_h = img.clone();


  int diagonal = sqrt((grad.rows*grad.rows)+(grad.cols*grad.cols));

  Mat houghSpace (diagonal*2,diagonal*2, CV_8UC1, Scalar(0)) ;

  cout<< endl << "begin lines" << endl;
  float angle;
  int P;
  int degrees = 0;
  //VOTING
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {

      if(grad.at<float>(y,x) == 255)
      {

        angle = dir.at<float>(y,x);
        for(int t= 0 ; t< 360 ; t++)
        {
          float radians = t*  (M_PI/180);
          P =  ( x*cos(radians) + y*sin(radians) ) + (diagonal);
          houghSpace.at<float> (P,t) ++;
        }

      }
    }
  }




  imshow("Hough Space LINES ", houghSpace);
  imwrite("Hough Space LINES.jpg", houghSpace);
  cout<< endl << "theta values" << endl;

  /*
  Mat intersection (grad.rows, grad.cols, CV_8UC1, Scalar(0)) ;
  float rho,theta;
  cout << "  " << grad.rows << "--" << grad.cols <<endl ;


  for(int i=0 ; i < houghSpace.rows ; i++)
  {
    //cout<< endl << "theta values" << endl;
    for(int j=0 ; j < houghSpace.cols ; j++)
    {
      if(houghSpace.at<float>(i,j) > 10)
      {
        rho = i;
        theta = j-180;
        theta = theta/180 * M_PI;
        for(int x = 0 ; x < intersection.cols ; x++)
        {
            int y =  cvRound(  ((-x) / tan(theta) ) + ( rho / sin(theta))  );

            if(y >= 0 && y < intersection.rows){
              cout << "  " << y << "," << x  ;
              intersection.at<float>(y,x) ++;
            }
        }
      }
    }
  }

  printf("\nintersection");
  imshow("Intersection Space LINES ", intersection);

  int maxi = 0; int max_X,max_Y;
  for(int y = 0 ; y < intersection.rows ; y++)
  {
    for(int x = 0 ; x < intersection.cols ; x++)
    {
      if (intersection.at<float>(y,x) > maxi){
        maxi = intersection.at<float>(y,x);
    //imwrite("Gradient Magnitude.jpg", grad);    max_Y = y;
        max_X = x;
      }
    }
  }

  circle(img_h, Point(max_X,max_Y), 2, cvScalar(0,255,0), 2);
  circle(img_h, Point(max_X,max_Y), 6, cvScalar(0,255,0), 2);

  */
  return img_h;


}

//*************************************************************************************************************************************************************
Mat threshold(Mat grad)
{
  float pix;
  float pixTh = 70;
  Mat thresh = grad.clone();
  for (int y = 0; y < grad.rows; y++){
    for (int x = 0; x < grad.cols; x++){
      pix = grad.at<float>(y,x);
      thresh.at<float>(y,x) = (pix > pixTh)? 255 : 0;
    }
  }
  //imshow("Gradient Magnitude after thresholding", thresh);
  imwrite("Threshold.jpg", thresh);
  return thresh;

}

Mat gradient(Mat image)
{
  // set up tranform kernel

  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);
  Mat grad = xs.clone();
  //looping through to get gradient Magnitude
  float dx,dy;
  for (int y = 0; y < xs.rows; y++){
    for (int x = 0; x < xs.cols; x++){
      dx = xs.at<float>(y,x);
      dy = ys.at<float>(y,x);

      grad.at<float>(y,x) = sqrt((dx*dx)+(dy*dy));
      //cout << "  " << grad.at<float>(y,x) ;
    }//cout<<endl<< "new line"<<endl;
  }
  normalize(grad, grad, 0, 255, NORM_MINMAX);
  //imwrite("Gradient Magnitude.jpg", grad);
  return grad;
}

Mat direction (Mat image)
{
  // set up tranform kernel
  Mat xKernel = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat yKernel = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
  Mat ys = convolution(image, yKernel);
  Mat xs = convolution(image, xKernel);

  Mat theta = xs.clone();
  float dx,dy, ang;

  for (int y = 0; y < xs.rows; y++){
    for (int x = 0; x < xs.cols; x++){
      dx = xs.at<float>(y,x);
      dy = ys.at<float>(y,x);
      theta.at<float>(y,x) = atan2(dy,dx);// in radians
      //cout << "  " << theta.at<float>(y,x) ;
    }//cout<<endl<< "new line"<<endl;
  }
  //imwrite("AngleRadians.jpg", theta);
  return theta;

}

Mat convolution(Mat base_img, Mat kernel){

  Mat new_img = base_img.clone(); //create output image
  new_img.convertTo(new_img, CV_32FC1); //convert to float so we can get larger numbers

  //Loop through the image
  for (int y = 0; y < base_img.rows; y++){
    for (int x = 0; x < base_img.cols; x++){
      float sum = 0;//reset sum at the start of a new pixel
      //Loop through the kernel
      for (int i = -1; i <= 1; i++){
        for (int j = -1; j <= 1; j++){
          //check if you go out of image range
          if (!(y+i < 0 || y+i > base_img.rows || x+j < 0 || x+j > base_img.cols)){
            sum += (base_img.at<uchar>(y+i,x+j) * kernel.at<float>(i+1,j+1));
          }
        }
      }//set the sum to the new image
      new_img.at<float>(y,x) = sum;
    }
  }
  return new_img;
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
