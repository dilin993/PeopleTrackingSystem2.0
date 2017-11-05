#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include "DataAssociation.h"
#include "BGSDetector.h"

using namespace std;
using  namespace cv;


#define TRACK_INIT_TH 0.9
#define REJ_TOL 30
#define WIDTH 640
#define HEIGHT 480




int main(int argc, const char * argv[])
{
    VideoCapture cap;
    Detector *detector = new BGSDetector();
    DataAssociation A(TRACK_INIT_TH, REJ_TOL,WIDTH,HEIGHT);

    if(argc<2)
    {
        cout << "Using webcam for input." << endl;
        cap = VideoCapture(1);
    }
    else
    {
        cout << "Using file: " << argv[1] << " for input." << endl;
        cap = VideoCapture(argv[1]);
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

    namedWindow("video capture", CV_WINDOW_AUTOSIZE);

    if (!cap.isOpened())
        return -1;

    Mat img;

    while (true)
    {
        cap >> img;
        if (!img.data)
            continue;

        vector<Rect> detections = detector->detect(img);
        vector<Point> detectionPoints;
        for(const auto r: detections)
        {
            detectionPoints.emplace_back(r.x+r.width/2,
                                         r.y+r.height/2);
        }
        A.assignTracks(detectionPoints,detector->histograms);

        vector<ParticleFilterTracker> &tracks = A.getTracks();

        for(int i=0;i<detections.size();i++)
        {
            rectangle(img, detections[i].tl(), detections[i].br(), cv::Scalar(200,10,10), 2);
        }

        for(int i=0;i<tracks.size();i++)
        {
            drawMarker(img, tracks[i].getPos(),
                       tracks[i].color,
                       MarkerTypes::MARKER_CROSS, 30, 10);
        }

        imshow("video capture", img);
        if (waitKey(20) >= 0)
            break;
    }
}