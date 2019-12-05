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
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

       // 1. Read Input Image
	Mat frame = imread(argv[1], 1);
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	/* Add in ground truths
	*     dart 7 = Point(340,190),Point(440,280)
	*			dart 13 =Point(420,130),Point(520,260)
	*/

	//int x = 320;
	//int y = 80;
	//rectangle( frame , Point(x,y),
	//				 Point(x + 70,y + 70),
	//				 Scalar( 255, 60, 60 ),
	//        2);
	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

// This function takes all the faces found by viola-jones, and does IOU with the ground truths
// it then returns the IOU value as a float
vector<float> iouVal(vector<Rect> faces){
	float inter;		//store intersect area
	vector<float> IOUs;	//store the IOU values of a face compared to all the truths
	vector<Rect> truths; //store the ground truth co-ordinates

	vector<float> bestIOUs;		//store the best result for a give face vs truth

	// declare the ground truths
	truths.push_back(Rect(340,190,100,90));
	for ( int j = 0; j < faces.size(); j++ ){	//loop through the truth boundries
		IOUs.clear();
		for( int i = 0; i < truths.size(); i++ )		// loop through the generated boundries
		{
			float xA = max(faces[j].x, 										truths[i].x);
			float yA = max(faces[j].y, 										truths[i].y);
			float xB = min(faces[j].x + faces[i].width, 	truths[i].x + truths[j].width);
			float yB = min(faces[j].y + faces[i].height, 	truths[i].y + truths[j].height);

			inter = max(0.0f, xB - xA + 1) * max(0.0f, yB - yA + 1);

			float areaA = (faces[j].x + faces[j].width - faces[j].x + 1) * (faces[j].y + faces[j].height - faces[j].y + 1);
			float areaB = (truths[i].x + truths[i].width - truths[i].x + 1) * (truths[i].y + truths[i].height - truths[i].y + 1);

			IOUs.push_back(inter / float(areaA + areaB - inter));
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
	for (int i = 0; i < acc.size(); i++){
		printf("%f accuracy\n", acc[i]);
	}
}
