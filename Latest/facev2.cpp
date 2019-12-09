/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

int ***malloc3dArray(int dim1, int dim2, int dim3);

Mat threshold(Mat grad);
Mat gradient(Mat image);
Mat direction (Mat image);
Mat convolution (Mat base_img, Mat kernel);

Mat houghCircles(Mat grad, Mat dir, Mat img);



/** Global variables */
String cascade_name = "frontalface.xml";
vector<Rect> truths;
vector<Rect> detect;
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

       // 1. Read Input Image
	Mat frame = imread(argv[1], 1);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	/*
	// Add in ground truths
	//truths.push_back(Rect(190,200,260-190,290-200));	//img 0
	// img 1
	// img 2
	// img 3
	// img 4
	// img 5
	// truths.push_back(Rect(66,139,52,64));
	// truths.push_back(Rect(55,249,57,70));
	// truths.push_back(Rect(194,216,54,68));
	// truths.push_back(Rect(254,169,48,60));
	// truths.push_back(Rect(295,243,48,67));
	// truths.push_back(Rect(383,192,52,55));
	// truths.push_back(Rect(430,235,52,68));
	// truths.push_back(Rect(518,182,46,55));
	// truths.push_back(Rect(563,244,51,72));
	// truths.push_back(Rect(650,196,46,50));
	// truths.push_back(Rect(682,246,49,64));
	//

	// truths.push_back(Rect(290,120,320-290,160-120)); //img 6
	// truths.push_back(Rect(350,195,415-350,280-195)); //img 7
	// truths.push_back(Rect(185,300,230-185,350-300)); //img 7
	// truths.push_back(Rect(90,180,190-90,350-180)); //img 9*/
	//truths.push_back(Rect(410,100,540-410,260-100)); //img 9


	Mat img = frame.clone();
	Mat img2=img.clone();

  Mat gray;
  cvtColor( img, gray, CV_BGR2GRAY );
  //GaussianBlur(gray, gray, Size(9 ,9 ),0 ,0 );

  Mat grad = gradient(gray);      //pixel range 0-255
  Mat angle = direction(gray);    //in radians
  Mat thresh = threshold(grad);   //pixel either 0 or 255
  Mat hcircle = houghCircles(thresh, angle, img2);
  grad.convertTo(grad,CV_8UC1);
  angle.convertTo(angle,CV_8UC1);
  hcircle.convertTo(hcircle,CV_8UC1);

  imshow("hough circle", hcircle);

  imwrite("Threshold.jpg", thresh);
  imwrite("Hough Circles.jpg", hcircle);


	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	for (int i = 0; i < truths.size(); i++){
		int x = truths[i].x;	int y = truths[i].y;
		rectangle( frame , Point(x,y),Point(x + truths[i].width,y + truths[i].height),Scalar( 255, 255, 255 ),2);
	}

	//rectangle( frame , Point(x,y),Point(x + truths[i].width,y + truths[i].height),Scalar( 30, 30, 255 ),2);


	// 4. Save Result Image
	imshow("Final conclusion", frame);
	imwrite( "detected.jpg", frame );
	waitKey(0);

	return 0;
}

// This function takes all the faces found by viola-jones, and does IOU with the ground truths
// it then returns the IOU value as a float
void iouVal(vector<Rect> faces){
	vector<float> IOUs;	//store the IOU values of a face compared to all the truths

	vector<float> bestIOUs;		//store the best result for a give face vs truth
	float maxArea=0.0,Area;
	int indexT,indexF;
	for ( int j = 0; j < faces.size(); j++ ){	//loop through the truth boundries
		IOUs.clear();
		for( int i = 0; i < truths.size(); i++ )		// loop through the generated boundries
		{
			Rect inter = faces[j] & truths[i];	//get intersection
			Rect unions = faces[j] | truths[i];	//get union
			//printf("Inter area: %d.  Union area: %d.   IOU: %f\n", inter.area(), unions.area(), (float)inter.area()/(float)unions.area());
			Area = (float)inter.area() / (float)unions.area();
			if(Area> maxArea)
			{
				maxArea = Area;
				indexT = i ;//index of truths[i]
				indexF = j ;//index of faces[j]
			}

			IOUs.push_back(Area);
	 	}

		bestIOUs.push_back(*max_element(IOUs.begin(),IOUs.end()));
	}

	int TP = 0;	//count true positives
	int FP = 0;	//count flase positives
	//get recall (TPR)
	for (int i = 0; i < bestIOUs.size(); i++){
		if (bestIOUs[i] > 0.4){	//if UOI val is > 0.5 count as true positive
			TP++;
			printf("True P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		}else {
			FP++;
			printf("False P found @ (%d,%d,%d,%d): %f accuracy\n", bestIOUs[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		}

	}

	float recall = (float)TP/truths.size();
	float prec	= (float)TP / ((float)TP + (float)FP);

	float FOne = 2.0f * ((prec * recall) / (prec + recall));

	printf("Recall [TPR] = %f\n",recall);
	printf("precision = %f\n",prec);
	printf("F1 score = %f\n",FOne);
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );
	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CASCADE_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	// for( int i = 0; i < faces.size(); i++ )
	// {
	// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	// }

	iouVal(faces);


}



//*************************************************************************************************************************************************************
/*
Mat houghCircles( Mat img){

	Mat img_h = img.clone();
	Mat gray  = img.clone();
  cvtColor( img, gray, CV_BGR2GRAY );
	Mat magni = gradient(gray);		//pixel range 0-255
	Mat dir = direction(gray);		//in radians
	Mat grad = threshold(magni);	//pixel either 0 or 255


  // Voting Threshold * * *
  int voteTh = 10 ;

  // Radius Range * * *
  int rmin = 60, rmax = 140;

  // declaring accumulator
  int ***acc; acc = malloc3dArray(grad.rows, grad.cols, rmax);

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
  imshow("Hough 2d", hough2D);          //displaying hough space
  imwrite("Hough 2d.jpg", hough2D);     //storing hough space
  hough2D.release();

	img_h.convertTo(img_h,CV_8UC1);
  imshow("hough circle", img_h);
	imwrite("Hough Circles.jpg", img_h);
  return img_h;
}
*/

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
			//hough2D.at<float>(y,x) = 50;
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
	circle(img_h, Point(max_X,max_Y), 2, cvScalar(255,0,255), 2);

	int ry = max_Y - max_R;
	int rx = max_X - max_R;
	int rht = max_R*2;
	int rwt = max_R*2;
	truths.push_back(Rect(rx,ry,rht,rwt)); //strongest circle


  //normalize(hough2D, hough2D, 0, 255, NORM_MINMAX);

  hough2D.convertTo(hough2D,CV_8UC1);   //Roundind off pixel values
  //std::cout << hough2D << '\n';         //printing hough space pixel values
  imshow("Hough 2d", hough2D);          //displaying hough space
  imwrite("Hough 2d.jpg", hough2D);     //storing hough space
  hough2D.release();
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
