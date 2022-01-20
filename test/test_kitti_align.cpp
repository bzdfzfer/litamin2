#include <iostream>
#include <thread>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include "dataloader/dataset.hpp"

#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/impl/fast_gicp_impl.hpp"
#include "litamin2/litamin2point2voxel.hpp"
#include "litamin2/litamin2point2voxelnewton.hpp"

using namespace litamin;
using namespace std;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: test_kitti_align path_to_kitti_dataset dataset_type out_path" << std::endl;
    return 0;
  }
  DatasetOptions dataset_options;

  // loading base path
  dataset_options.root_path = argv[1];
  // loading dataset type
  std::string dataset_type = argv[2];
  if(dataset_type == "KITTI_raw")
  	dataset_options.dataset = KITTI_raw;  	
  else if(dataset_type == "KITTI")
  	dataset_options.dataset = KITTI;

  // loading dataset sequence id.
  std::string out_path = argv[3];

  auto sequences = get_sequences(dataset_options);
  int num_sequences = (int) sequences.size();
  cout << "sequences num: " << num_sequences << endl;

  // use downsample_resolution=1.0 for fast registration
  double downsample_resolution = 0.25;
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);


  litamin::LiTAMIN2Point2Voxel<pcl::PointXYZ, pcl::PointXYZ> litamin2;
  // litamin::LiTAMIN2Point2VoxelNewton<pcl::PointXYZ, pcl::PointXYZ> litamin2;
  litamin2.setNumThreads(4);
  litamin2.setResolution(3.0);
  litamin2.setMaxCorrespondenceDistance(1.0);
  litamin2.setTransformationEpsilon(1e-2);
  litamin2.setMaximumIterations(64);

  // trajectory for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory(new pcl::PointCloud<pcl::PointXYZ>);
  trajectory->push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));

  pcl::visualization::PCLVisualizer vis;
  vis.setBackgroundColor(0, 0, 0);
  vis.addCoordinateSystem(1.0);
  vis.addText("KITTI trajectory pcl_visualizer", 10, 10, "debugger text", 0);
  vis.initCameraParameters();
  // position x,y,z         view x,y,z      view up: x,y,z
  vis.setCameraPosition(99.8088, 142.249, 533.837,  177.075, 20.2209, 21.9058,  0.986978, 0.101531, 0.124763);
  vis.setCameraClipDistances(519.902, 554.931);  
  vis.addPointCloud<pcl::PointXYZ>(trajectory, "trajectory");


  for(int i=0; i < num_sequences; i++) {
  	int sequence_id = sequences[i].sequence_id;
  	cout << "current sequence_id: " << sequence_id << endl;
    auto iterator_ptr = get_dataset_sequence(dataset_options, sequence_id);
	  
    iterator_ptr->SetInitFrame(0);
	  iterator_ptr->printDatasetType(dataset_options);

    // set initial frame as target
    int frame_id = 0;
    voxelgrid.setInputCloud(iterator_ptr->Frame(frame_id).makeShared());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.filter(*target);
    litamin2.setInputTarget(target);

    // sensor pose sequence
    int seq_size = iterator_ptr->NumFrames();
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(seq_size);
    poses[0].setIdentity();

    while(iterator_ptr->HasNext() ) {
      frame_id ++;
    	pcl::PointCloud<pcl::PointXYZ> frame = iterator_ptr->Next();
      cout << "Seq.Frame_id: [" << sequence_id << "] - " << frame_id << endl; 
    	// cout << "frame pts num: " << frame.size() << endl;

      // set the current frame as source
      voxelgrid.setInputCloud(frame.makeShared());
      pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
      voxelgrid.filter(*source);
      litamin2.setInputSource(source);

      // align and swap source and target cloud for next registration
      pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
      litamin2.align(*aligned);      
      litamin2.swapSourceAndTarget();

      // accumulate pose
      poses[frame_id] = poses[frame_id - 1] * litamin2.getFinalTransformation().cast<double>();

      // visualization
      trajectory->push_back(pcl::PointXYZ(poses[frame_id](0, 3), poses[frame_id](1, 3), poses[frame_id](2, 3)));
      vis.updatePointCloud<pcl::PointXYZ>(trajectory, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(trajectory, 255.0, 0.0, 0.0), "trajectory");
      vis.spinOnce();      
    }
    
    // save the estimated poses
    std::string out_file_path;
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << sequence_id << ".txt";
    out_file_path = out_path + ss.str();
    std::ofstream ofs(out_file_path);
    for (const auto& pose : poses) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
          if (i || j) {
            ofs << " ";
          }

          ofs << pose(i, j);
        }
      }
      ofs << std::endl;        
    }


  }

  return 0;

}