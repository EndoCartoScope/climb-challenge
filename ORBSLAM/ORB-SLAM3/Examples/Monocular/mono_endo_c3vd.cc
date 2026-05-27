/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include <regex>

#include<opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#include"System.h"

using namespace std;
namespace fs = boost::filesystem;

std::map<int, std::pair<std::string, std::string>>
getImagePairs(const std::vector<std::string> &paths, const std::string &colorSuffix, const std::string &depthSuffix, int &initial_number_frame) {
    std::map<int, std::pair<std::string, std::string>> imagePairs;
    std::regex reColor(R"((\d+)\.png)");
    std::regex reDepth(R"((\d+)_depth\.tiff)");
    std::smatch match;
    initial_number_frame = 10e6;
    for (const auto &path: paths) {
        for (const auto &entry: fs::directory_iterator(path)) {
            if (!fs::is_regular_file(entry.status())) continue;

            std::string name = entry.path().filename().string();
            int number = -1;

            // Verificar si el archivo es de color
            if (std::regex_match(name, match, reColor) && match.size() > 1) {
                number = std::stoi(match.str(1));
                imagePairs[number].first = entry.path().string();
                if (number < initial_number_frame) {
                    initial_number_frame = number;
                }
            }
            //     // Verificar si el archivo es de profundidad
            // else if (std::regex_match(name, match, reDepth) && match.size() > 1) {
            //     number = std::stoi(match.str(1));
            //     imagePairs[number].second = entry.path().string();
            // }
        }
        std::cout << "[MonoDepthEndoC3VD::getImagePairs]: Found " << imagePairs.size() << " color images in " << path << std::endl;
    }

    return imagePairs;
}

void extractInterImage(const cv::Mat &orig_frame_1, cv::Mat &ext_image, cv::Mat &aux_image) {
    cv::Size new_size(orig_frame_1.cols / 2, orig_frame_1.rows / 2);
    int index_aux = 0;
    for (int i = 0; i < orig_frame_1.rows; i+=2, index_aux++)
    {
        orig_frame_1.row(i).copyTo(aux_image.row(index_aux));
    }

    cv::resize(aux_image, ext_image, new_size);
}


int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_endo_c3vd path_to_vocabulary path_to_settings path_to_video" << endl;
        return 1;
    }
    
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    cv::FileNode node = fSettings["System"]["imageEndFile"];
    string colorSuffix = "_color.png";
    string depthSuffix = "_depth.tiff";
    if (!node.empty() && node.isString()) {
        colorSuffix = node.string();
    }
    node = fSettings["Camera"]["numFramesAvoid"];
    int nFramesAvoided = 0;
    if (!node.empty() && node.isInt()) {
        nFramesAvoided = node.operator int();
    }
    if (nFramesAvoided < 0) {
        std::cerr << "*Camera.numFramesAvoid is less than 0, it must be a positive value*" << std::endl;
        return -1;
    }
    node = fSettings["Camera"]["fps"];
    int nFps = 30;
    if (!node.empty() && node.isInt()) {
        nFps = node.operator int();
    }
    
    int numVideos = argc - 3;
    // Retrieve paths to images
    std::vector<std::string> vImagePaths;
    for (int i = 0; i < numVideos; ++i) {
        vImagePaths.push_back(argv[3 + i]);
    }
    int initial_number_frame = 0;
    auto imagePairs = getImagePairs(vImagePaths, colorSuffix, depthSuffix, initial_number_frame);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(imagePairs.size());

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << imagePairs.size() << endl << endl;

    // Main loop
    cv::Mat orig_frame;
    int numImages = 0;
    for (size_t i = 0; i < imagePairs.size(); ++i) {

        orig_frame = cv::imread(imagePairs[i].first, cv::IMREAD_UNCHANGED);
        if (orig_frame.empty()) {
            std::cerr << "[MonoDepthEndoC3VD::Main]: Could not open or find the image: " << imagePairs[i].first << std::endl;
            return -1;
        }
      
        if (nFramesAvoided > 0) {
            if (i % (nFramesAvoided + 1) != 0) {
                // It avoids the amount of frames selected
                continue;
            }
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        
        cv::Mat ext_frame_d = cv::Mat (orig_frame.rows/2, orig_frame.cols/2, orig_frame.type());
        cv::Mat aux_frame_d = cv::Mat (orig_frame.rows/2, orig_frame.cols, orig_frame.type());
        extractInterImage(orig_frame, ext_frame_d, aux_frame_d);

        // Pass the image to the SLAM system
        int vid_pos_frame = i;
        double vid_sec = (float(1) / nFps) * vid_pos_frame;
        string name = imagePairs[i].first;
        name = name.substr(name.find_last_of("/") + 1);
        cout << "Processing frame: " << name << endl;

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(ext_frame_d,vid_sec,vector<ORB_SLAM3::IMU::Point>(), name);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack.push_back(ttrack);

        // Wait to load the next frame
        if (ttrack < (float(1) / nFps))
            usleep(((float(1) / nFps) - ttrack) * 1e6);
        numImages++;
    }

    // Stop all threads
    SLAM.Shutdown();
    string folder_name = "./output/camera_trajectory/";
    SLAM.SaveTrajectoryAllMaps(folder_name);
    folder_name = "./output/3D_maps/";
    SLAM.SaveInColmapFormatMaps3D(folder_name);

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<vTimesTrack.size(); ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[vTimesTrack.size()/2] << endl;
    cout << "mean tracking time: " << totaltime/vTimesTrack.size() << endl;    

    return 0;
}
