#include <iostream>
#include <thread>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include "dataloader/dataset.hpp"

using namespace litamin;
using namespace std;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "usage: path_to_kitti_dataset dataset_type" << std::endl;
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


  auto sequences = get_sequences(dataset_options);
  int num_sequences = (int) sequences.size();
  cout << "sequences num: " << num_sequences << endl;
  pcl::visualization::PCLVisualizer vis;


  for(int i=0; i < num_sequences; i++) {
  	int sequence_id = sequences[i].sequence_id;
  	cout << "current sequence_id: " << sequence_id << endl;
    auto iterator_ptr = get_dataset_sequence(dataset_options, sequence_id);
	iterator_ptr->SetInitFrame(0);
	iterator_ptr->printDatasetType(dataset_options);
    int frame_id = 0;
    while(iterator_ptr->HasNext() && !vis.wasStopped()) {
    	pcl::PointCloud<pcl::PointXYZ> frame = iterator_ptr->Next();
    	cout << "frame pts num: " << frame.size() << endl;

    	// clear first.
    	vis.removeShape("raw_cloud");
    	
		vis.addPointCloud<pcl::PointXYZ>(frame.makeShared(), 
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(frame.makeShared(), 
											255.0, 255.0, 255.0), 
			"raw_cloud");
		 vis.spinOnce (100);
		 // std::this_thread::sleep_for(100ms);
    }
  }

  return 0;

}