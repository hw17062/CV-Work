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

	int x = 416;
	int y = 115;
	rectangle( frame , Point(x,y),
					 Point(x + 117,y + 147),
					 Scalar( 30, 30, 255 ),
	        2);
	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

// This function takes all the faces found by viola-jones, and does IOU with the ground truths
// it then returns the IOU value as a float
vector<float> iouVal(vector<Rect> faces){
	vector<float> IOUs;	//store the IOU values of a face compared to all the truths
	vector<Rect> truths; //store the ground truth co-ordinates

	vector<float> bestIOUs;		//store the best result for a give face vs truth

	// declare the ground truths
	truths.push_back(Rect(416,115,117,147));
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
		if (acc[i] > 0.5){	//if UOI val is > 0.5 count as true positive
			TP++;
			printf("TP found: %f accuracy\n", acc[i]);
		}else {
			FP++;
			printf("FP found: %f accuracy\n", acc[i]);
		}

	}
	float recall = (float)TP/(float)faces.size();
	float prec	= (float)TP / ((float)TP + (float)FP);

	float FOne = 2.0f * ((prec * recall) / (prec + recall));

	printf("Recall [TPR] = %f\n",recall);
	printf("precision = %f\n",prec);
	printf("F1 score = %f\n",FOne);

}
