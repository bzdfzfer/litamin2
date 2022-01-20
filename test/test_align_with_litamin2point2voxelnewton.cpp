#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>


#include <fast_gicp/gicp/fast_gicp.hpp>
#include "fast_gicp/gicp/impl/fast_gicp_impl.hpp"
#include <litamin2/litamin2point2voxelnewton.hpp>



// benchmark for PCL's registration methods
template <typename Registration>
void test_pcl(Registration& reg, 
              const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, 
              const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  std::cout << "source_cloud size: " << source->size() << std::endl;
  std::cout << "target_cloud size: " << target->size() << std::endl;

  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "align result: \n" << reg.getFinalTransformation() << std::endl;
  std::cout << "single:" << single << "[msec] " << std::endl;

  // // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++) {
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "100times:" << multi << "[msec] fitness_score:" << fitness_score << std::endl;
}

// benchmark for fast_gicp registration methods
template <typename Registration>
void test(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  std::cout << "source_cloud size: " << source->size() << std::endl;
  std::cout << "target_cloud size: " << target->size() << std::endl;
  
  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  reg.clearTarget();
  reg.clearSource();
  reg.setInputTarget(target);
  reg.setInputSource(source);
  reg.align(*aligned);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  std::cout << "align result: \n" << reg.getFinalTransformation() << std::endl; 
  std::cout << "single:" << single << "[msec] " << std::endl;

  // // 100 times
  // t1 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 100; i++) {
  //   reg.clearTarget();
  //   reg.clearSource();
  //   reg.setInputTarget(target);
  //   reg.setInputSource(source);
  //   reg.align(*aligned);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double multi = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "100 times:" << multi << "[msec] " << std::flush;

  // std::cout << "  , average time: " << multi/100 << "[msec] "<< std::endl;

  // // for some tasks like odometry calculation,
  // // you can reuse the covariances of a source point cloud in the next registration
  // t1 = std::chrono::high_resolution_clock::now();
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_ = target;
  // pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_ = source;
  // for (int i = 0; i < 100; i++) {
  //   reg.swapSourceAndTarget();
  //   reg.clearSource();

  //   reg.setInputTarget(target_);
  //   reg.setInputSource(source_);
  //   reg.align(*aligned);

  //   target_.swap(source_);
  // }
  // t2 = std::chrono::high_resolution_clock::now();
  // double reuse = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;

  // std::cout << "100 times_reuse:" << reuse << "[msec] fitness_score:" << fitness_score << std::endl;
}

/**
 * @brief main
 */
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "usage: gicp_align target_pcd source_pcd" << std::endl;
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (pcl::io::loadPCDFile(argv[1], *target_cloud)) {
    std::cerr << "failed to open " << argv[1] << std::endl;
    return 1;
  }
  if (pcl::io::loadPCDFile(argv[2], *source_cloud)) {
    std::cerr << "failed to open " << argv[2] << std::endl;
    return 1;
  }
  int a = 0;
  // remove invalid points around origin
  source_cloud->erase(
    std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    source_cloud->end());
  target_cloud->erase(
    std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    target_cloud->end());

  // downsampling
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*filtered);
  target_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*filtered);
  source_cloud = filtered;

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" << std::endl;

  std::cout << "--- LiTAMIN2Point2VoxelNewton_test ---" << std::endl;
  litamin::LiTAMIN2Point2VoxelNewton<pcl::PointXYZ, pcl::PointXYZ> litamin2_test;
  // fast_gicp uses all the CPU cores by default
  litamin2_test.setNumThreads(4);
  litamin2_test.setResolution(3.0);
  litamin2_test.setMaxCorrespondenceDistance(1.0);
  litamin2_test.setTransformationEpsilon(1e-2);
  litamin2_test.setMaximumIterations(64);
  test(litamin2_test, target_cloud, source_cloud);

  // std::cout << "--- fgicp_ceres ---" << std::endl;
  // fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_ceres;
  // // fast_gicp uses all the CPU cores by default
  // fgicp_ceres.setNumThreads(8);
  // fgicp_ceres.setTransformationEpsilon(1e-2);
  // fgicp_ceres.setMaxCorrespondenceDistance(0.5);
  // fgicp_ceres.setLocalParameterization(true);
  // fgicp_ceres.setLSQType(fast_gicp::LSQ_OPTIMIZER_TYPE::CeresDogleg);
  // test(fgicp_ceres, target_cloud, source_cloud);

  std::cout << std::endl << std::endl << "Visualizing ......." << std::endl;


  pcl::visualization::PCLVisualizer vis;
  vis.initCameraParameters();
  vis.setCameraPosition(15.5219, 6.13405, 22.536,   8.258, -0.376825, -0.895555,    0.0226091, 0.961419, -0.274156);
  vis.setCameraFieldOfView(0.523599);
  vis.setCameraClipDistances(0.00522511, 50); 

  vis.addPointCloud<pcl::PointXYZ>(source_cloud, 
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(source_cloud, 255.0, 255.0, 255.0), 
    "source");
  vis.addPointCloud<pcl::PointXYZ>(target_cloud, 
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(source_cloud, 0.0, 255.0, 0.0), 
    "target");

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
  litamin2_test.align(*aligned);
  vis.addPointCloud<pcl::PointXYZ>(aligned, 
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(aligned, 0.0, 0.0, 255.0), 
    "aligned");


  vis.spin();

  return 0;
}
