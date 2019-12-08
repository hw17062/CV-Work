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

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
vector<Rect> truths;
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

       // 1. Read Input Image
	Mat frame = imread(argv[1], 1);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// Add in ground truths
	// truths.push_back(Rect(420,0,210,220));   //img0
	// truths.push_back(Rect(165,100,255,250));   //img1
	// truths.push_back(Rect(90,85,110,110));     //img2
	// truths.push_back(Rect(310,135,90,95));		 //img3
	// truths.push_back(Rect(155,65,380-155,335-65)); //img4
	// truths.push_back(Rect(415,125,550-415,270-125));	//img 5
	// truths.push_back(Rect(205,110,280-205,190-110));	//img 6
	// truths.push_back(Rect(230,150,410-230,335-140));	//img 7
	// truths.push_back(Rect(830,200,970-830,350-200));	//img 8
	// truths.push_back(Rect(60,240,130-60,350-240));	//img 8
	// truths.push_back(Rect(170,15,465-170,320-15));	//img 9
	// truths.push_back(Rect(80,90,195-80,230-90));	//img 10
	// truths.push_back(Rect(575,120,640-575,220-120));	//img 10
	// truths.push_back(Rect(914,140,955-914,220-130));	//img 10
	// truths.push_back(Rect(165,95,240-165,180-95));	//img 11
	// truths.push_back(Rect(150,60,220-150,230-60));	//img 12
	// truths.push_back(Rect(255,100,420-255,270-100));	//img 13
	// truths.push_back(Rect(105,85,260-105,240-85));	//img 14
	// truths.push_back(Rect(970,80,1125-970,235-80));	//img 14
	truths.push_back(Rect(120,35,300-120,210-35));	//img 15


	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	for (int i = 0; i < truths.size(); i++){

		int x = truths[i].x;
		int y = truths[i].y;

		rectangle( frame , Point(x,y),
						 Point(x + truths[i].width,y + truths[i].height),
						 Scalar( 30, 30, 255 ),
		        2);
		}

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

// This function takes all the faces found by viola-jones, and does IOU with the ground truths
// it then returns the IOU value as a float
vector<float> iouVal(vector<Rect> faces){
	vector<float> IOUs;	//store the IOU values of a face compared to all the truths

	vector<float> bestIOUs;		//store the best result for a give face vs truth


	for ( int j = 0; j < faces.size(); j++ ){	//loop through the truth boundries
		IOUs.clear();
		for( int i = 0; i < truths.size(); i++ )		// loop through the generated boundries
		{
			Rect inter = faces[j] & truths[i];	//get intersection
			Rect unions = faces[j] | truths[i];	//get union
			//printf("Inter area: %d.  Union area: %d.   IOU: %f\n", inter.area(), unions.area(), (float)inter.area()/(float)unions.area());
			IOUs.push_back((float)inter.area() / (float)unions.area());
	 	}
		bestIOUs.push_back(*max_element(IOUs.begin(),IOUs.end()));
	}

	return bestIOUs;
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
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	vector<float> acc = iouVal(faces);

	int TP = 0;	//count true positives
	int FP = 0;	//count flase positives
	//get recall (TPR)
	for (int i = 0; i < acc.size(); i++){
		if (acc[i] > 0.4){	//if UOI val is > 0.5 count as true positive
			TP++;
			printf("True P found @ (%d,%d,%d,%d): %f accuracy\n", acc[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		}else {
			FP++;
			printf("False P found @ (%d,%d,%d,%d): %f accuracy\n", acc[i],faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		}

	}

	float recall = (float)TP/truths.size();
	float prec	= (float)TP / ((float)TP + (float)FP);

	float FOne = 2.0f * ((prec * recall) / (prec + recall));

	printf("Recall [TPR] = %f\n",recall);
	printf("precision = %f\n",prec);
	printf("F1 score = %f\n",FOne);

}
