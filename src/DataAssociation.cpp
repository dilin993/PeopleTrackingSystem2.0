//
// Created by dilin on 8/13/17.
//

#include "DataAssociation.h"

DataAssociation::DataAssociation(double TRACK_INIT_TH,int REJ_TOL,
                                int WIDTH,int HEIGHT):
        TRACK_INIT_TH(TRACK_INIT_TH),
        REJ_TOL(REJ_TOL),
        WIDTH(WIDTH),
        HEIGHT(HEIGHT)
{

}



DataAssociation::~DataAssociation()
{
}

vector<ParticleFilterTracker> &DataAssociation::getTracks()
{
    return tracks;
}

void DataAssociation::assignTracks(vector<Point> detections,
                                   vector<MatND> histograms)
{
    vector<vector<double>> costMat;

    for(int i=0;i<detections.size();i++)
    {
        if(tracks.size()==0) // initialize the first track
        {
            tracks.emplace_back(detections[i],histograms[i],
                                sigma_propagate);
        }
        else // calculate cost matrix
        {
            double min_e = DBL_MAX;
            vector<double> row(tracks.size());
            row.clear();
            for (int j = 0; j < tracks.size(); j++)
            {
                tracks[j].update();
                double e = averageError(detections[i],tracks[j].getPos(),
                                        histograms[i],tracks[j].histogram);
                if (e < min_e)
                    min_e = e;
                row.push_back(e);
            }
            if (min_e > TRACK_INIT_TH) // initialize new track
            {
                ParticleFilterTracker tr(detections[i],histograms[i],
                                         sigma_propagate);
                tracks.emplace_back(tr);
                double e = averageError(detections[i],tr.getPos(),
                                        histograms[i],tr.histogram);
                row.push_back(e);
            }
            costMat.push_back(row);
        }
    }


    // data association using hungarian algorithm
    if(costMat.size()>0)
    {
        HungarianAlgorithm hungarianAlgorithm;
        vector<int> assignment;
        double cost = hungarianAlgorithm.Solve(costMat,assignment);
        vector<bool> hasAssigned;

        // initialize hasAssigned to false
        for(int j=0;j<tracks.size();j++)
        {
            hasAssigned.push_back(false);
        }

        // assign detections
        for(int i=0;i<assignment.size();i++)
        {
            int j = assignment[i];
            if(j>=0)
            {
                tracks[j].assignDetection(detections[i],histograms[i]);
                hasAssigned[j] = true;
            }
        }

        // delete unnecessary tracks
        cout << "current tracks: " << endl;
        for(int j=0;j<tracks.size();j++)
        {
            tracks[j].updateAssociation(hasAssigned[j]);
            State state = tracks[j].getState();
            double v = sqrt(pow(state.vx,2.0)+pow(state.vy,2.0));
            cout << "[" << j << "]";
            cout << "\tx=" << state.x;
            cout << " y=" << state.y;
            cout << " vx=" << state.vx;
            cout << " vy=" << state.vy;
            cout << " v=" << v;
            cout << " assigned=" << hasAssigned[j] << endl;

            if(v>VEL_TH ||
                    state.x > WIDTH ||
                    state.y > HEIGHT ||
                    state.x < 0 ||
                    state.y < 0 ||
               tracks[j].consectiveInvisibleCount > REJ_TOL)
            {
                cout << "deleted track : " << j << " , size : " << tracks.size() << endl;
                //Tracker *tr = tracks[j];
                tracks.erase(tracks.begin()+j);
                //delete tr;
                j--;
            }
        }

    }
}

double DataAssociation::averageError(Point a, Point b)
{
    double de = sqrt(pow((double)(a.x-b.x),2.0) + pow((double)(a.y-b.y),2.0));
    return de;
}

double DataAssociation::averageError(Point a, Point b, MatND histA, MatND histB)
{
    double dh = compareHist(histA,histB,HISTCMP_CORREL);
    dh = 20.0 / (dh+1e-10);
    double e = dh + averageError(a,b);
    return  e;
}

void DataAssociation::setSize(int width, int height)
{
    WIDTH = width;
    HEIGHT = height;
}
