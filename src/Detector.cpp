//
// Created by dilin on 8/11/17.
//

#include "Detector.h"

cv::MatND Detector::getHistogram(cv::Mat img, cv::Rect region)
{
    cv::Mat rgb_base;
    //cv::Mat hsv_base;
    rgb_base = img(region);
   // cv::cvtColor(rgb_base, hsv_base, cv::COLOR_BGR2HSV);

    cv::MatND hist_base;
    /// Using 50 bins for hue and 60 for saturation
    int histSize[] = { 8, 8, 8 };

    // hue varies from 0 to 179, saturation,rgb from 0 to 255
//    float h_ranges[] = { 0, 180 };
//    float s_ranges[] = { 0, 256 };
    float r_ranges[] = { 0, 256 };
    float g_ranges[] = { 0, 256 };
    float b_ranges[] = { 0, 256 };

    const float* ranges[] = {  r_ranges, g_ranges, b_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1, 2};

    cv::Mat images[] = {rgb_base};


    calcHist( images, 1, channels, cv::Mat(), hist_base, 3, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    return hist_base;
}

cv::MatND Detector::getHistogram(cv::Mat img, cv::Rect region, cv::Mat mask)
{
    cv::Mat rgb_base,shape;
    //cv::Mat hsv_base;
    rgb_base = img(region);
    shape = mask(region);
    // cv::cvtColor(rgb_base, hsv_base, cv::COLOR_BGR2HSV);

    cv::MatND hist_base;
    /// Using 50 bins for hue and 60 for saturation
    int histSize[] = { 8, 8, 8 };

    // hue varies from 0 to 179, saturation,rgb from 0 to 255
//    float h_ranges[] = { 0, 180 };
//    float s_ranges[] = { 0, 256 };
    float r_ranges[] = { 0, 256 };
    float g_ranges[] = { 0, 256 };
    float b_ranges[] = { 0, 256 };

    const float* ranges[] = {  r_ranges, g_ranges, b_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1, 2};

    cv::Mat images[] = {rgb_base};


    calcHist( images, 1, channels, shape, hist_base, 3, histSize, ranges, true, false );


    cv::FileStorage file("../hist.ext", cv::FileStorage::WRITE);
    file << "histogram" << hist_base;
    file << "bbox" << region;

    cv::imwrite("../image.jpg",img);
    cv::imwrite("../mask.jpg",mask);

    file.release();

    normalize( hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    return hist_base;
}
