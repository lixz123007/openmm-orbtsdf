/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <unistd.h>
#include "cuda_runtime.h"
#include "demo.cuh"
#include <Python.h>
#include<vector>
int mi=0;

namespace ORB_SLAM2
{
// extern "C" 
// void integrate(float * cam_K, float * cam2base, float * depth_im,
//                int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
//                float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
//                float * voxel_grid_TSDF, float * voxel_grid_weight);
System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)), mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
    //重建
    voxel_grid_origin_x = -0.250f; // Location of voxel grid origin in base frame camera coordinates
    voxel_grid_origin_y = -0.250f;
    voxel_grid_origin_z = 0.01f;
    voxel_size = 0.002f;
    trunc_margin = voxel_size * 5;
    voxel_grid_dim_x = 500;
    voxel_grid_dim_y = 500;
    voxel_grid_dim_z = 500;
    voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    voxel_grid_rgb = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
    
    for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
        voxel_grid_TSDF[i] = 1.0f;
        //voxel_grid_rgb[i]=0f;}
    memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    memset(voxel_grid_rgb, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
    
    cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    cudaMalloc(&gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
    
    //checkCUDA(__LINE__, cudaGetLastError());
    cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_voxel_grid_rgb, voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
    
    //checkCUDA(__LINE__, cudaGetLastError());
    float cam_K[3 * 3];
    cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    cam_K[0]=fx;
    cam_K[1]=0;
    cam_K[2]=cx;
    cam_K[3]=0;
    cam_K[4]=fy;
    cam_K[5]=cy;
    cam_K[6]=0;
    cam_K[7]=0;
    cam_K[8]=1;
    
    cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
    cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

//initial python!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // Py_Initialize();

    // //判断初始化是否成功
    // if(!Py_IsInitialized())
    // {
    //     printf("Python init failed!\n");
    //     //return -1;
    // }
    
    


    
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp,const int times)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
        
    }
    }
    cout<<"1.";
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);
    if(!Tcw.data){
        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
        return Tcw;
    }
    // cv::Mat output_image;
    // cvtColor(im, output_image, CV_BGR2RGB);
    // string savelocation="/home/finch/catkin_ws/src/ORB_SLAM2/datas/";
    // savelocation=savelocation+to_string(mi)+".jpg";

        ////cv::imwrite("/home/finch/SLAM/src/orbtsdf-openmm/b1.jpg", im);

    //cvSaveImage(im, pSrc);
    //python on!
        // PyRun_SimpleString("import sys");
        // //PyRun_SimpleString("sys.path.append('../')");//若python文件在c++工程下
        // // 添加python文件所在的位置
        // PyRun_SimpleString("sys.path.append('/home/finch/SLAM/src/orbtsdf-openmm/')");
        // PyObject* pModule = NULL;
        // PyObject* pFunc = NULL;
        
        // //导入python文件
        // pModule = PyImport_ImportModule("callmask");
        // if (!pModule) {
        //     printf("Can not open python file!\n");
        //     //return -1;
        // }

    
    //pFunc = PyObject_GetAttrString(pModule, "printHello");
        // pFunc = PyObject_GetAttrString(pModule, "masktry");
        // PyObject* result=PyObject_CallObject(pFunc, NULL);
        // char *pout = PyBytes_AsString (result);
        // //char *s=pout;
        // int a[99][6]={0};
        // int b[im.rows][im.cols];
        // int ii=0;
        // int ij=0;
        // int ik=0;
    //long  d=0;

    
    // while(pout[ik]){
        
    // if(isdigit(pout[ik]))  {
 
    //   b[ii][ij]=b[ii][ij]*10+(int)(pout[ik]-'0');
    //     //printf("%d  %c %d\n",a[ii][ij],pout[ik],(int)pout[ik]);
    //     //bboxes[i][4]*100,labels[i],bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3]
    //     //(accuracy>80,labels,x1,y1,x2,y2)
	//  }
    // if(pout[ik]==',')ij++;
    // if(ij>im.cols){ii++;ij=0;}
    //  ik++;}

        // while(pout[ik]){
            
        // if(isdigit(pout[ik]))  {
    
        //   a[ii][ij]=a[ii][ij]*10+(int)(pout[ik]-'0');
        //     //printf("%d  %c %d\n",a[ii][ij],pout[ik],(int)pout[ik]);
        //     //bboxes[i][4]*100,labels[i],bboxes[i][0],bboxes[i][1],bboxes[i][2],bboxes[i][3]
        //     //(accuracy>80,labels,x1,y1,x2,y2)
        //  }
        // if(pout[ik]==' ')ij++;
        // if(pout[ik]==','){ii++;ij=0;}
        //  ik++;}



    cv::Mat black;
    
        // mi++;
        // unique_lock<mutex> lock2(mMutexState);
        // mTrackingState = mpTracker->mState;
        // mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        // mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        // printf("real start\n");
        // return Tcw;
    
    //cvtColor(im, output_image, CV_BGR2RGB);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //black=cv::imread("/home/finch/data/black.jpg", 0);
    // if(times==1)
    // {
    // black=cv::imread("/home/finch/data/depth/"+to_string(times)+".jpg", 1);}
    // else
    // black=cv::imread("/home/finch/data/depth/"+to_string(times-1)+".jpg", 1);

    //cout<<im.cols<<im.rows<<black.cols<<black.rows;

    float * gpu_cam2base;
    float * gpu_depth_im;
    float * gpu_rgb;
    //cv::Mat cim = im;
 
    // if(mbReset){
    //     return Tcw;
    // }
    // int cr2=256*256;
    // int cr=256;
    cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
    cudaMalloc(&gpu_depth_im, depthmap.rows * depthmap.cols* sizeof(float));
    //checkCUDA(__LINE__, cudaGetLastError());
    cudaMalloc(&gpu_rgb, depthmap.rows * depthmap.cols* sizeof(float));
    //cout<<"2.";
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat mOw = -Rcw.t()*tcw;
    //cout<<"3.";
    cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    mOw.copyTo(Twc.rowRange(0,3).col(3));
    //cout<<"5.";    
    float cam2base[]={Twc.at<float>(0,0),Twc.at<float>(0,1),Twc.at<float>(0,2),Twc.at<float>(0,3),
    Twc.at<float>(1,0),Twc.at<float>(1,1),Twc.at<float>(1,2),Twc.at<float>(1,3),
    Twc.at<float>(2,0),Twc.at<float>(2,1),Twc.at<float>(2,2),Twc.at<float>(2,3),
    Twc.at<float>(3,0),Twc.at<float>(3,1),Twc.at<float>(3,2),Twc.at<float>(3,3) };
    cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    //cout<<Twc<<endl;
    float depth[depthmap.rows * depthmap.cols];
    float rgb[depthmap.rows * depthmap.cols];
    /*int c05=0;
    int c051=0;
    int c12=0;
    int c23=0;
    int c3=0;*/


    for (int r = 0; r <depthmap.rows; ++r)
    for (int c = 0; c <depthmap.cols ; ++c) {
     //5-12mask//if(black.ptr<uchar>(r)[c]!=0){continue;}//||black.at<Vec3b>(r, c)[1]!=0||black.at<Vec3b>(r, c)[2]!=0)continue;//mask way to do!cout<<r<<"r c: "<<c<<" result:"<<black.ptr<uchar>(r)[c]<<endl;
       // if(b[r, c]!=0){continue;}//||black.at<Vec3b>(r, c)[1]!=0||black.at<Vec3b>(r, c)[2]!=0)continue;//mask way to do!

      depth[r * depthmap.cols + c] = (float)(depthmap.at<unsigned short>(r, c)) /5000.0f;
     // if(mbRGB)
      rgb[r * depthmap.cols + c] = im.ptr(r, c)[2]+im.ptr(r, c)[1]*256+im.ptr(r, c)[0]*256*256;//colour
      //else
      //rgb[r * depthmap.cols + c] = im.ptr(r, c)[2]+im.ptr(r, c)[1]*256+im.ptr(r, c)[0]*256*256;
      /*if (depth[r *depthmap.cols+ c] < 0.5f) c05++;
      if (depth[r *depthmap.cols+ c] < 1.0f&&depth[r *depthmap.cols+ c] >= 0.5f) c051++;
      if (depth[r *depthmap.cols+ c] < 2.0f&&depth[r *depthmap.cols+ c] >= 1.0f) c12++;
      if (depth[r *depthmap.cols+ c] < 3.0f&&depth[r *depthmap.cols+ c] >= 2.0f) c23++;
      if (depth[r *depthmap.cols+ c] >= 3.0f) c3++;
      */


      if (depth[r *depthmap.cols+ c] > 6.0f) // Only consider depth < 6m
       {
           //cout<<"depth[r *depthmap.cols+ c] "<<depth[r *depthmap.cols+ c] <<endl;
          depth[r * depthmap.cols + c] = 0; 
       } 
    }

        //ij=0;
    
    
    //use boxxing way to clean things awawy!!!!!!!!!!!!!111

    // while(a[ij][1]==0||a[ij][1]==15)//label=0 means person 
    // {
    //     if(a[ij][0]<80){ij++;break;}//accuracy=80!!!!!!1
    //     for (int r = a[ij][3]; r <a[ij][5]; ++r)
    //         for (int c = a[ij][2]; c <a[ij][4] ; ++c) {
    //              depth[r * depthmap.cols + c] =0;
    //         }
    //     ij++;
    // }
    //bye!!


    //cout<<c05<<" 0.5 0.5-1: "<<c051<<" 1-2: "<<c12<<" 2-3:"<<c23<<"3+:"<<c3<<endl;
 //if(mi>1){
    cudaMemcpy(gpu_depth_im, depth, depthmap.rows * depthmap.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_rgb, rgb, depthmap.rows * depthmap.cols * sizeof(float), cudaMemcpyHostToDevice);
    
    //checkCUDA(__LINE__, cudaGetLastError());
    //demo();
    //cout<<Twc<<"gpu_cam_k"<<gpu_cam2base;
    integrate(gpu_rgb,gpu_cam_K, gpu_cam2base, gpu_depth_im,
    depthmap.rows, depthmap.cols, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
    voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
    gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,gpu_voxel_grid_rgb);
    
    //mi++;
    cout<<"times:"<<mi++;
  //cv::imwrite("/home/finch/catkin_ws/src/ORB_SLAM2/blackback.jpg", black);



    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}
void System::SaveMap(){
  cout<<"saving map: ";  
    cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cout<<"1.....";
  cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cout<<"2..... ";
  cudaMemcpy(voxel_grid_rgb, gpu_voxel_grid_rgb, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
   cout<<"3......";
   float tsdf_thresh=0.2, weight_thresh=0.0;
   int num_pts = 0;
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh)
      num_pts++;

    cout<<"points:"<<num_pts;
  // Create header for .ply file
  FILE *fp = fopen("map.ply", "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_pts);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {

    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh) {
      // Compute voxel indices in int for higher positive number range
      //cout<<"i: "<<i<<"\t";
      int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
      int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
      int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);
        float old_a=voxel_grid_rgb[i];
        uchar old_b=(uchar)(old_a/256/256);
        uchar old_g=(uchar)((old_a-old_b*256*256)/256);
        uchar old_r=(uchar)(old_a-old_b*256-old_g*256*256);
      //cout<<"i: "<<i<<"\t";
      // Convert voxel indices to float, and save coordinates to ply file
      float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
      float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
      float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
      fwrite(&pt_base_x, sizeof(float), 1, fp);
      fwrite(&pt_base_y, sizeof(float), 1, fp);
      fwrite(&pt_base_z, sizeof(float), 1, fp);
      fwrite(&old_r, sizeof(uchar), 1, fp);
      fwrite(&old_g, sizeof(uchar), 1, fp);
      fwrite(&old_b, sizeof(uchar), 1, fp);
      //cout<<"i: "<<i<<"\t";
    }
  }
  fclose(fp);

  std::ofstream outFile("map.bin", std::ios::binary | std::ios::out);
  float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
  float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
  float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
  outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
  outFile.write((char*)&voxel_size, sizeof(float));
  outFile.write((char*)&trunc_margin, sizeof(float));
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
  outFile.close();



}
cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
