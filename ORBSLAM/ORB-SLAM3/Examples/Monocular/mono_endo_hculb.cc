#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>
#include "System.h"
#include "Converter.h"


using namespace std;

int main(int argc, char **argv)
{
    // Init phase starts here: everything before the per-frame video-feed loop.
    auto t_main_start = std::chrono::steady_clock::now();

    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_endo_hclb path_to_vocabulary path_to_settings path_to_video " << endl;
        return 1;
    }

    cv::Mat frame;
    cv::VideoCapture vid_cap;
    if(!vid_cap.open(string(argv[3])))
    {
        cerr << "Error: Video file " << argv[3] << " doesn't exist" << endl;
        return -1;
    }

    int fpsVideo = vid_cap.get(cv::CAP_PROP_FPS);
    int numFramesVideo = vid_cap.get(cv::CAP_PROP_FRAME_COUNT);
    double videoDuration_ms = (((float)numFramesVideo) / fpsVideo) * 1e3;
    string strVideoDuration = "Video duration";

    string started = "Endomapper/";
    string sequence = string(argv[3]);
    int index = sequence.find(started);
    sequence = sequence.substr(index + started.size());
    sequence = sequence.substr(0, sequence.find_first_of("/"));
    std::cout << "Sequence: " << sequence << std::endl;
    std::vector<string> vSequences;
    vSequences.push_back(sequence);

    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);

    cv::FileNode node  = fSettings["Camera"]["numFramesAvoid"];

    int nFramesAvoided = 0;
    if(!node.empty() && node.isInt())
    {
        nFramesAvoided = node.operator int();
    }
    if(nFramesAvoided < 0)
    {
        std::cerr << "*Camera.numFramesAvoid is less than 0, it must be a positive value*" << std::endl;
        return -1;
    }

    node = fSettings["Camera"]["fps"];
    int nFps = 50;
    if(!node.empty() && node.isInt())
    {
        nFps = node.operator int();
    }

    // Get initial time deffined in setting file
    int initial_second = 0;
    int initial_minute = 0;
    node = fSettings["System"]["initialTimeSecond"];
    if(!node.empty() && node.isInt())
    {
        initial_second = node.operator int();
    }

    node = fSettings["System"]["initialTimeMinute"];
    if(!node.empty() && node.isInt())
    {
        initial_minute = node.operator int();
    }

    if(initial_second < 0 || initial_second >= 60)
    {
        std::cerr << "*System.initialTimeSecond has an incorrect value: " << initial_second << ". It must be >= 0 and < 60*" << std::endl;
        return -1;
    }

    if(initial_minute < 0)
    {
        std::cerr << "*System.initialTimeMinute has an incorrect value: " << initial_minute << ". It must be >= 0*" << std::endl;
        return -1;
    }

    // Get initial time deffined in setting file
    int finish_second = 0;
    int finish_minute = 0;
    node = fSettings["System"]["finishTimeSecond"];
    if(!node.empty() && node.isInt())
    {
        finish_second = node.operator int();
    }

    node = fSettings["System"]["finishTimeMinute"];
    if(!node.empty() && node.isInt())
    {
        finish_minute = node.operator int();
    }

    if(finish_second < 0 || finish_second >= 60)
    {
        std::cerr << "*System.finishTimeSecond has an incorrect value: " << finish_second << ". It must be >= 0 and < 60*" << std::endl;
        return -1;
    }

    double vid_initial_time_millisec = 0;
    if(initial_second != 0 || initial_minute != 0)
    {
        vid_initial_time_millisec = ((double)((initial_minute * 60) + initial_second)) *1e3;
        std::cout << "Sequence initial time is " << std::setw(2) << std::setfill('0') << initial_minute << ":" << initial_second << std::endl;
        std::cout << " which in millisecons is: " << vid_initial_time_millisec << std::endl;
    }

    double vid_finish_time_millisec = 0;
    if(finish_second != 0 || finish_minute != 0)
    {
        vid_finish_time_millisec = ((double)((finish_minute * 60) + finish_second)) *1e3;
        if(vid_initial_time_millisec > vid_finish_time_millisec)
        {
            std::cout << "Invalid finish time, initial time is greater than the finish time... Hence, finish time is ignored" << std::endl;
            vid_finish_time_millisec = 0;
        }
        else
        {
            std::cout << "Sequence finish time is " << std::setw(2) << std::setfill('0') << finish_minute << ":" << finish_second << std::endl;
            std::cout << " which in millisecons is: " << vid_finish_time_millisec << std::endl;
        }
    }

    int initial_frame = 0;
    int finish_frame = numFramesVideo;
    node = fSettings["System"]["initialNumberFrame"];
    if(!node.empty() && node.isInt())
    {
        initial_frame = node.operator int();
        if(initial_frame < 0)
        {
            std::cerr << "*System.initialNumberFrame has an incorrect value: " << initial_frame << ". It must be >= 0*" << std::endl;
            return -1;
        }
    }

    node = fSettings["System"]["finishNumberFrame"];
    if(!node.empty() && node.isInt())
    {
        finish_frame = node.operator int();
        if(finish_frame < 0)
        {
            std::cerr << "*System.finishNumberFrame has an incorrect value: " << finish_frame << ". It must be >= 0*" << std::endl;
            return -1;
        }
        else if (finish_frame == 0)
        {
            finish_frame = numFramesVideo;
        }
    }

    if(initial_frame != 0 && finish_frame != 0 && finish_frame < initial_frame)
    {
        std::cout << "Invalid finish frame number, initial frame number is greater than the finish one... Hence, finish frame number is ignored" << std::endl;
        finish_frame = 0;
    }
    else if (initial_frame != 0)
    {
        std::cout << "Sequence goes from " << initial_frame << " to " << finish_frame << " frames" << std::endl;
    }

    // Vector for tracking time statistics
    vector<double> vTimesTrack;

    cout << endl << "-------" << endl;
    cout.precision(17);

    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale();
    int nFrames = 0;

    double vid_millisec = vid_cap.get(cv::CAP_PROP_POS_MSEC);
    int vid_pos_frame = vid_cap.get(cv::CAP_PROP_POS_FRAMES);

    bool bUseNumber = false;
    bool bUseTime = false;

    if(initial_frame > 0)
    {
        vid_cap.set(cv::CAP_PROP_POS_FRAMES, initial_frame);
        bUseNumber = true;
    }
    else if(vid_initial_time_millisec > 0)
    {
        vid_cap.set(cv::CAP_PROP_POS_MSEC, vid_initial_time_millisec);
        bUseTime = true;
    }

    int num_frames = 0;

    const double target_fps = static_cast<double>(nFps);
    const auto frame_period =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(1000.0 / target_fps));

    auto next_tick = std::chrono::steady_clock::now();

    // Processing phase: from the first frame read to the last pose decided.
    // Only this interval feeds the ranking metric.
    auto t_processing_start = std::chrono::steady_clock::now();

    bool bReadFrame = vid_cap.read(frame);
    while(bReadFrame)
    {
        next_tick += frame_period;
        vid_millisec = vid_cap.get(cv::CAP_PROP_POS_MSEC);
        vid_pos_frame = vid_cap.get(cv::CAP_PROP_POS_FRAMES);

        if(bUseNumber)
        {
            if(finish_frame > 0 && vid_pos_frame >= finish_frame)
            {
                    break;
            }
        }
        else if(bUseTime)
        {
            if(vid_finish_time_millisec > 0 && vid_millisec >= vid_finish_time_millisec)
            {
                    break;
            }
        }

        if(nFramesAvoided > 0)
        {
            if(vid_pos_frame % (nFramesAvoided + 1) != 0)
            {
                // It avoids the amount of frames selected
                bReadFrame = vid_cap.read(frame);
                continue;
            }
        }

        if(frame.empty())
        {
            std::cerr << "Frame lost..." << std::endl;
            bReadFrame = vid_cap.read(frame);
            continue;
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        num_frames++;

        // Pass the image to the SLAM system
        double vid_seconds = vid_millisec * 1e-3;
        string name = to_string(vid_pos_frame);

        SLAM.TrackMonocular(frame, vid_seconds, vector<ORB_SLAM3::IMU::Point>(), name);

        auto ttrack = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();


        vTimesTrack.push_back(ttrack);
        nFrames++;
        std::this_thread::sleep_until(next_tick);
        if (std::chrono::steady_clock::now() > next_tick + frame_period) {
            next_tick = std::chrono::steady_clock::now();
        }
        bReadFrame = vid_cap.read(frame);
    }

    // End of processing phase. Everything below (Shutdown + Save*) is post-processing
    // and is excluded from both init_seconds and processing_seconds.
    auto t_processing_end = std::chrono::steady_clock::now();

    // Stop all threads
    SLAM.Shutdown();
    string folder_name = "./output/camera_trajectory/";
    SLAM.SaveTrajectoryAllMaps(folder_name);
    folder_name = "./output/3D_maps/";
    SLAM.SaveInColmapFormatMaps3D(folder_name);

    // Per-run timing artifact. ./output/ exists by now because Save*() above creates it.
    {
        const double init_seconds =
            std::chrono::duration<double>(t_processing_start - t_main_start).count();
        const double processing_seconds =
            std::chrono::duration<double>(t_processing_end - t_processing_start).count();
        std::ofstream runtime_file("./output/runtime.txt");
        runtime_file << std::fixed << std::setprecision(6);
        runtime_file << "init_seconds=" << init_seconds << "\n";
        runtime_file << "processing_seconds=" << processing_seconds << "\n";
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<vTimesTrack.size(); ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nFrames/2] << endl;
    cout << "mean tracking time: " << totaltime/nFrames << endl;

    return 0;
}
