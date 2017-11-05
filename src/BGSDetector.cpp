//
// Created by dilin on 10/23/17.
//

#include "BGSDetector.h"

std::vector<cv::Rect> BGSDetector::detect(cv::Mat &img)
{
    std::vector<cv::Rect> found,detections;
    histograms.clear();

    for(int i=2;i>0;i--)
    {
        frames[i] = frames[i-1].clone();
    }
    frames[0] = img.clone();

    if(frameCount<3)
    {
        frameCount++;
        return detections;
    }

    backgroundSubstraction(frames[0],frames[1],frames[2],
                           bgModel,mask,TH);

//    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
//    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
//    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

    /*
    cv::dilate(imgThresh, imgThresh, structuringElement7x7);
    cv::erode(imgThresh, imgThresh, structuringElement3x3);
    */
    cv::Mat maskPost;
    cv::dilate(mask,maskPost, structuringElement);
    cv::dilate(maskPost, maskPost, structuringElement);
    cv::erode(maskPost, maskPost, structuringElement);



    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(maskPost, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > convexHulls(contours.size());

    for (unsigned int i = 0; i < contours.size(); i++)
    {
        cv::convexHull(contours[i], convexHulls[i]);
    }

    for (auto &convexHull : convexHulls) {
        Blob possibleBlob(convexHull);

        if (possibleBlob.currentBoundingRect.area() > 100 &&
            possibleBlob.dblCurrentAspectRatio >= 0.2 &&
            possibleBlob.dblCurrentAspectRatio <= 1.25 &&
            possibleBlob.currentBoundingRect.width > 20 &&
            possibleBlob.currentBoundingRect.height > 20 &&
            possibleBlob.dblCurrentDiagonalSize > 30.0 &&
            (cv::contourArea(possibleBlob.currentContour) /
             (double)possibleBlob.currentBoundingRect.area()) > 0.40)
        {
            found.push_back(possibleBlob.currentBoundingRect);
        }
    }

    size_t i, j;

    for (i=0; i<found.size(); i++)
    {
        cv::Rect r = found[i];
        for (j=0; j<found.size(); j++)
            if (j!=i && (r & found[j])==r)
                break;
        if (j==found.size())
        {
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
            detections.push_back(r);
            histograms.push_back(getHistogram(img,r));//,mask));
        }

    }

    return detections;
}


void BGSDetector::backgroundSubstraction(cv::Mat &frame0, cv::Mat &frame1, cv::Mat &frame2
        , cv::Mat &bgModel, cv::Mat &mask, double TH)
{
    cv::Mat frame0g,frame1g,frame2g;

   // convert frames to gray
    cvtColor(frame0,frame0g,cv::COLOR_BGR2GRAY);
    cvtColor(frame1,frame1g,cv::COLOR_BGR2GRAY);
    cvtColor(frame2,frame2g,cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(frame0g,frame0g,cv::Size(5, 5), 0);
    cv::GaussianBlur(frame1g,frame1g,cv::Size(5, 5), 0);
    cv::GaussianBlur(frame2g,frame2g,cv::Size(5, 5), 0);

    bgModel = 0.1*frame0g + 0.2*frame1g + 0.7*frame2g;

    cv::Mat diff;
    absdiff(frame0g,bgModel,diff);

    threshold(diff,mask,TH,255,cv::THRESH_BINARY);
}

BGSDetector::BGSDetector(double TH) :
TH(TH)
{
    frameCount = 0;
    pMOG2 = cv::createBackgroundSubtractorMOG2(10);
}

Blob::Blob(std::vector<cv::Point> _contour)
{
    currentContour = _contour;

    currentBoundingRect = cv::boundingRect(currentContour);

    cv::Point currentCenter;

    currentCenter.x = (currentBoundingRect.x + currentBoundingRect.x + currentBoundingRect.width) / 2;
    currentCenter.y = (currentBoundingRect.y + currentBoundingRect.y + currentBoundingRect.height) / 2;

    centerPositions.push_back(currentCenter);

    dblCurrentDiagonalSize = sqrt(pow(currentBoundingRect.width, 2) + pow(currentBoundingRect.height, 2));

    dblCurrentAspectRatio = (float)currentBoundingRect.width / (float)currentBoundingRect.height;

    blnStillBeingTracked = true;
    blnCurrentMatchFoundOrNewBlob = true;

    intNumOfConsecutiveFramesWithoutAMatch = 0;
}
