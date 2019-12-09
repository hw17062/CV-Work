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
Mat printPixel(Mat image);

Mat threshold(Mat grad);
Mat gradient(Mat image);
Mat direction (Mat image);
Mat convolution (Mat base_img, Mat kernel);

Mat houghCircles(Mat grad, Mat dir, Mat img);
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
  Mat hcircle = houghCircles(thresh, angle, img2);

  Mat hline = houghLines(thresh, angle, img2);



  grad.convertTo(grad,CV_8UC1);
  angle.convertTo(angle,CV_8UC1);
  hcircle.convertTo(hcircle,CV_8UC1);
  //hline.convertTo(hline,CV_8UC1);
  //imshow("Gradient Magnitude", grad);
  //imshow("Gradient Angle", angle);
  imshow("hough circle", hcircle);
  //imshow("hough line", hline);
  waitKey(0);

  //imwrite("Gradient Magnitude.jpg", grad);
  imwrite("Threshold.jpg", thresh);
  //imwrite("AngleRadians.jpg", angle);
  imwrite("Hough Circles.jpg", hcircle);
  //imwrite("Hough line.jpg", hline);

  return 0;
}

//*************************************************************************************************************************************************************

Mat houghCircles(Mat grad, Mat dir, Mat img){

  // Voting Threshold * * *
  int voteTh = 10 ;

  // Radius Range * * *
  int rmin = 60, rmax = 140;

  // declaring accumulator
  int ***acc; acc = malloc3dArray(grad.rows, grad.cols, rmax);
  Mat img_h = img.clone();
  int a,b,c,d;

  //INTILIALISING accumulator
  for(int y = 0 ; y < grad.rows ; y++){
    for(int x = 0 ; x < grad.cols ; x++){
      for(int r = 0 ; r < rmax ; r++){
        acc[y][x][r] = 0;}}}

  //---------------------------------------------------------------
  //VOTING
  float angle = 0;
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {
      if (grad.at<float>(y,x) == 255)
      {
        angle = dir.at<float>(y,x);
        for(int r = rmin ; r < rmax ; r++)
        {
          a = (int)(y - r * sin(angle));
          b = (int)(x - r * cos(angle));//cos is linked with cols or x-axis of cartesion plane
          c = (int)(y + r * sin(angle));
          d = (int)(x + r * cos(angle));
          //VOTING -------
          if(a>0 && a<grad.rows && b>0 && b<grad.cols){  acc[a][b][r] ++;}
          if(c>0 && c<grad.rows && d>0 && d<grad.cols){  acc[c][d][r] ++;}
        }
      }
    }
  }

  //---------------------------------------------------------------
  // Creating image for 2D-HOUGH Space
  Mat hough2D(grad.rows, grad.cols, CV_32FC1, Scalar(0)) ;

  int vote = 0, radius = 0;
  int max_Vote = 0, max_Y = 0, max_X = 0, max_R = 0;

  //looping to (1)create 2d Hough (2)printing circles above threshold (3)finding Strongest centre
  for(int y = 0 ; y < grad.rows ; y++)
  {
    for(int x = 0 ; x < grad.cols ; x++)
    {
      vote=0; radius=0;
      for(int r = rmin ; r < rmax ; r++)
      {
        hough2D.at<float>(y,x) += acc[y][x][r];   //adding the votes to create 2D HOUGH space

        if(acc[y][x][r] > vote){   //Finding the radius with maximum votes in each point of the image
          vote = acc[y][x][r];
          radius = r;
        }
      }

      //We do this to avoid detecting multiple circles at the same point * * *
      if(vote > voteTh){  //Checking if vote is above the threshold
        circle(img_h, Point(x,y), radius, cvScalar(0,255,0), 2);   //Printing the circle
        circle(img_h, Point(x,y), 1, cvScalar(255,0,0), 2);        //Printing centre pt
      }

      if(vote > max_Vote){  //Storing the point with the maximim vote in the entire image
        max_Vote = vote;  max_R = radius; max_Y = y;  max_X = x;
      }

    }
  }

  //---------------------------------------------------------------

  //Printing the circle with MAXimum VOTE or the Strongest Centre
  cout << endl << "Maximum vote : " << max_Vote << " at " << max_Y << "," << max_X << "," << max_R << endl;
  circle(img_h, Point(max_X,max_Y), max_R, cvScalar(255,0,255), 2);

  //normalize(hough2D, hough2D, 0, 255, NORM_MINMAX);

  hough2D.convertTo(hough2D,CV_8UC1);   //Roundind off pixel values
  //std::cout << hough2D << '\n';         //printing hough space pixel values
  imshow("Hough 2d", hough2D);          //displaying hough space
  imwrite("Hough 2d.jpg", hough2D);     //storing hough space
  hough2D.release();
  return img_h;
}

//*************************************************************************************************************************************************************

Mat houghLines(Mat grad, Mat dir, Mat img)
{

  int voteTh = 2;


  Mat img_h = img.clone();


  int diagonal = sqrt((grad.rows*grad.rows)+(grad.cols*grad.cols));

  Mat houghSpace (diagonal, 180, CV_8UC1, Scalar(0)) ;
  //std::vector<Point> ACC;

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
        for(float t = angle - 0.3 ; t < angle + 0.3; t += 0.02)///////////////////////////
        {
          P =  round( x*cos(t) + y*sin(t) );
          //cout << "  " << P << "," << t ;
          degrees = (int)(t /M_PI * 180);
          degrees+=180;
          //P += diagonal;
          if( P >= 0 && P < diagonal)
            houghSpace.at<float>(P,degrees)++;

        }
      }
    }//cout<<endl<< "new line"<<endl;
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
        max_Y = y;
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
  imshow("Gradient Magnitude after thresholding", thresh);
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

Mat printPixel(Mat image){
  Mat copy = image.clone();
  //Mat copy2 = Mat(image.size(), CV_8UC1);

  for(int y =  0; y < copy.rows ; y++){
    cout << y << "}  ";
    for(int x =  0; x < copy.cols ; x++){
      cout << copy.at<float>(y,x) <<" : ";
    }cout << endl<<endl;
  }cout << endl<< endl;

  //imshow("copy2", copy2);
  //imwrite("copy2.jpg",copy2);
  return copy;

}
