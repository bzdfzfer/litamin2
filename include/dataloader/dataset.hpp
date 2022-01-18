#ifndef LITAMIN_DATASET_HPP
#define LITAMIN_DATASET_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <iostream>

namespace litamin {

using PointT = pcl::PointXYZ;

enum DATASET {
    KITTI_raw = 0,
    KITTI = 1,
};

struct DatasetOptions {

    DATASET dataset;

    std::string root_path;

    bool fail_if_incomplete = false; // Whether to fail if all sequences are not present on disk

    double min_dist_lidar_center = 3.0; // Threshold to filter points too close to the LiDAR center

    double max_dist_lidar_center = 100.0; // Threshold to filter points too far to the LiDAR center

    int nclt_num_aggregated_pc = 220; // The number of hits to aggregate for NCLT Dataset

};

class DatasetSequence {
public:
    virtual ~DatasetSequence() = 0;

    virtual bool HasNext() const = 0;

    virtual pcl::PointCloud<PointT> Next() = 0;

    virtual size_t NumFrames() const {
        return -1;
    }

    virtual pcl::PointCloud<PointT> Frame(size_t index) const {
        throw std::runtime_error("Random Access is not supported");
    }

    virtual void SetInitFrame(int frame_index) {
        init_frame_id_ = frame_index;
    };

    virtual bool WithRandomAccess() const {
        return false;
    }

    virtual void printDatasetType(const DatasetOptions& options) {
        switch (options.dataset) {
            case KITTI_raw:
                std::cout << "KITTI_raw" << std::endl;
                break;
            case KITTI:
                std::cout << "KITTI odometry" << std::endl;
                break;  
            }
    }
protected:
    int init_frame_id_ = 0; // The initial frame index of the sequence
};



struct SequenceInfo {

    std::string sequence_name;

    int sequence_id = -1;

    int sequence_size = -1;

};

// Returns the Pairs sequence_id, sequence_size found on disk for the provided options
std::vector<SequenceInfo> get_sequences(const DatasetOptions &);

// Reads a PointCloud from the Dataset KITTI
pcl::PointCloud<PointT> read_kitti_pointcloud(const DatasetOptions &, const std::string &path);

// Reads a PointCloud from the disk
pcl::PointCloud<PointT> read_pointcloud(const DatasetOptions &, int sequence_id, int frame_id);


// Returns the Sequence Name as a string given its id
std::string sequence_name(const DatasetOptions &, int sequence_id);


// Returns a DatasetSequence
std::shared_ptr<DatasetSequence> get_dataset_sequence(const DatasetOptions &, int sequence_id = -1);

}


#endif //LITAMIN_DATASET_HPP
