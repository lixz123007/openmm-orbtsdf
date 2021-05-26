#include "cuda_runtime.h"

#include <device_launch_parameters.h>
#include "demo.cuh"
#include "iostream"

// using namespace std;
// using std::cout;
// using std::fmin;

float floatmin(float a,float b)
{
  if(a>b)
  return b;
  else
  return a;
}
__global__
void Integrate(float * rgba,float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight,float * voxel_grid_rgb) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;
  
  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];
    //cout<<"pt_cam_z"<<pt_cam_z<<endl;
    if(pt_cam_z<0)pt_cam_z=-pt_cam_z;
    if (pt_cam_z <= 0)
      continue;
    
    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];
    float rgba_val =rgba[pt_pix_y * im_width + pt_pix_x];

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float diff = depth_val - pt_cam_z;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    
    int new_b=(int)(rgba_val/256/256);
    int new_g=(int)((rgba_val-new_b*256*256)/256);
    int new_r=rgba_val-new_g*256-new_b*256*256;
    float old_a=voxel_grid_rgb[volume_idx];
    int old_b=(int)(old_a/256/256);
    int old_g=(int)((old_a-old_b*256*256)/256);
    int old_r=old_a-old_g*256-old_b*256*256;
    new_b = min(255, (int)((weight_old*old_b + new_b) / weight_new));
    new_g = min(255, (int)((weight_old*old_g + new_g) / weight_new));
    new_r = min(255, (int)((weight_old*old_r + new_r) / weight_new));
    voxel_grid_rgb[volume_idx]=new_b*256*256 + new_g*256 + new_r;
    //int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    
    
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}

extern "C"
void integrate(float * rgba,float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight,float * voxel_grid_rgb){
                 Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(rgba,cam_K, cam2base, depth_im,im_height
                                                         , im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                         voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                         voxel_grid_TSDF, voxel_grid_weight,voxel_grid_rgb);
               }

