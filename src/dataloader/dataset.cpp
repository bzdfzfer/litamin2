#include "dataloader/dataset.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>
#include <fstream>

namespace litamin {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// HARD CODED VALUES FOR KITTI

const char *KITTI_raw_SEQUENCE_NAMES[] = {
		"2011_10_03/2011_10_03_drive_0027_sync/",
		"2011_10_03/2011_10_03_drive_0042_sync/",
		"2011_10_03/2011_10_03_drive_0034_sync/",
		"2011_09_26/2011_09_26_drive_0067_sync/",
		"2011_09_30/2011_09_30_drive_0016_sync/",
		"2011_09_30/2011_09_30_drive_0018_sync/",
		"2011_09_30/2011_09_30_drive_0020_sync/",
		"2011_09_30/2011_09_30_drive_0027_sync/",
		"2011_09_30/2011_09_30_drive_0028_sync/",
		"2011_09_30/2011_09_30_drive_0033_sync/",
		"2011_09_30/2011_09_30_drive_0034_sync/",	
};
const char *KITTI_SEQUENCE_NAMES[] = {
        "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"
};

const int KITTI_raw_SEQUENCE_IDS[] = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10};
const int KITTI_SEQUENCE_IDS[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

const int NUMBER_SEQUENCES_KITTI_raw = 10;
const int NUMBER_SEQUENCES_KITTI = 22;

const int LENGTH_SEQUENCE_KITTI[] = {4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200, 920, 1060,
                                     3280, 630, 1900, 1730, 490, 1800, 4980, 830, 2720};


//Specific Parameters for KITTI_raw and KITTI
const double KITTI_MIN_Z = -5.0; //Bad returns under the ground
const double KITTI_GLOBAL_VERTICAL_ANGLE_OFFSET = 0.205; //Issue in the intrinsic calibration of the KITTI Velodyne HDL64

inline PointT rotationCorrection(const PointT& raw_pt) {
	PointT corrected_pt = raw_pt;

	Eigen::Vector3d raw_pt_vec = raw_pt.getVector3fMap().template cast<double>();
	Eigen::Vector3d rotationVector = raw_pt_vec.cross(Eigen::Vector3d(0,0,1));
	rotationVector.normalize();
	Eigen::Matrix3d rotationScan;
	rotationScan = Eigen::AngleAxisd(KITTI_GLOBAL_VERTICAL_ANGLE_OFFSET *M_PI / 180., rotationVector);
	
	Eigen::Vector3d cor_pt_vec = rotationScan * raw_pt_vec;	

	corrected_pt.getVector3fMap() = cor_pt_vec.cast<float>();
	return corrected_pt;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Returns the Path to the folder containing a sequence's point cloud Data
inline std::string pointclouds_dir_path(const DatasetOptions &options, const std::string &sequence_name) {
    std::string folder_path = options.root_path;
    if (folder_path.size() > 0 && folder_path[folder_path.size() - 1] != '/')
        folder_path += '/';

    switch (options.dataset) {
        case KITTI_raw:
            folder_path += sequence_name + "/velodyne_points/data/";
            break;
        case KITTI:
            folder_path += sequence_name + "/velodyne/";
            break;
        default:
            throw std::runtime_error("Not Implemented!");
    };
    return folder_path;
}



inline std::string frame_file_name_kitti(int frame_id) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << frame_id;
    return  ss.str() + ".bin";
}

inline std::string frame_file_name_kitti_raw(int frame_id) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << frame_id;
    return ss.str() + ".bin";
}

inline std::string frame_file_name(const DatasetOptions &options, int frame_id)
{
    switch (options.dataset) {
        case KITTI_raw:
        	return frame_file_name_kitti_raw(frame_id);
        case KITTI:
        	return frame_file_name_kitti(frame_id);
    }	
    throw std::runtime_error("Dataset not recognised");
}
/* -------------------------------------------------------------------------------------------------------------- */
std::vector<SequenceInfo> get_sequences(const DatasetOptions &options) {
    // TODO Use a FileSystem library (e.g. C++17 standard library) / other to test existence of files
    std::vector<SequenceInfo> sequences;
    int num_sequences;
    switch (options.dataset) {
        case KITTI_raw:
            num_sequences = NUMBER_SEQUENCES_KITTI_raw;
            break;
        case KITTI:
            num_sequences = NUMBER_SEQUENCES_KITTI;
            break;
    }

    sequences.reserve(num_sequences);

    for (auto i(0); i < num_sequences; ++i) {
        SequenceInfo new_sequence_info;
        switch (options.dataset) {
            case KITTI_raw:
                new_sequence_info.sequence_id = KITTI_raw_SEQUENCE_IDS[i];
                new_sequence_info.sequence_size = LENGTH_SEQUENCE_KITTI[new_sequence_info.sequence_id] + 1;
                new_sequence_info.sequence_name = KITTI_raw_SEQUENCE_NAMES[new_sequence_info.sequence_id];
                break;
            case KITTI:
                new_sequence_info.sequence_id = KITTI_SEQUENCE_IDS[i];
                new_sequence_info.sequence_size = LENGTH_SEQUENCE_KITTI[new_sequence_info.sequence_id] + 1;
                new_sequence_info.sequence_name = KITTI_SEQUENCE_NAMES[new_sequence_info.sequence_id];
                break;
        }

        sequences.push_back(new_sequence_info);
    }
    return sequences;
}

/* -------------------------------------------------------------------------------------------------------------- */
std::string sequence_name(const DatasetOptions &options, int sequence_id) {
    switch (options.dataset) {
        case KITTI_raw:
        	return KITTI_raw_SEQUENCE_NAMES[sequence_id];
        case KITTI:
            return KITTI_SEQUENCE_NAMES[sequence_id];
    }
    throw std::runtime_error("Dataset not recognised");
}

/* -------------------------------------------------------------------------------------------------------------- */
pcl::PointCloud<PointT> read_pointcloud(const DatasetOptions &options, int sequence_id, int frame_id) {

    std::string frames_dir_path = pointclouds_dir_path(options, sequence_name(options, sequence_id));
    std::string frame_path = frames_dir_path + frame_file_name(options, frame_id);

    // Read the pointcloud
    switch (options.dataset) {
        case KITTI_raw:
        case KITTI:
            return read_kitti_pointcloud(options, frame_path);
    }
    throw std::runtime_error("Dataset not recognised");
}


/* -------------------------------------------------------------------------------------------------------------- */
pcl::PointCloud<PointT> read_kitti_pointcloud(const DatasetOptions &options, const std::string &path) {
    pcl::PointCloud<PointT> frame;

    // std::cout << "file path: " << path << std::endl;
    // read bin frame file.
    std::ifstream lidar_data_file(path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));

    PointT point;
    for(size_t i=0; i < lidar_data_buffer.size(); i += 4) {
    	point.x = lidar_data_buffer[i];
    	point.y = lidar_data_buffer[i+1];
    	point.z = lidar_data_buffer[i+2];

    	double r = point.getVector3fMap().template cast<double>().norm();
    	if ((r > options.min_dist_lidar_center) && (r < options.max_dist_lidar_center) &&
            (point.z > KITTI_MIN_Z)) {
 		   	frame.push_back(point);
		}
    }

    //Intrinsic calibration of the vertical angle of laser fibers 
    // (take the same correction for all lasers)
    for (int i = 0; i < (int) frame.size(); i++) {
        frame.points[i] = rotationCorrection(frame.points[i]);
    }
    return frame;
}


    
/* -------------------------------------------------------------------------------------------------------------- */
DatasetSequence::~DatasetSequence() =
default;

/* -------------------------------------------------------------------------------------------------------------- */


/// DirectoryIterator for KITTI_raw and KITTI
class DirectoryIterator : public DatasetSequence {
public:
    explicit DirectoryIterator(const DatasetOptions &options, int sequence_id = -1) : options_(options),
                                                                                      sequence_id_(
                                                                                              sequence_id) {
        switch (options.dataset) {
            case KITTI_raw:
                num_frames_ = LENGTH_SEQUENCE_KITTI[sequence_id] + 1;
                break;
            case KITTI:
                num_frames_ = LENGTH_SEQUENCE_KITTI[sequence_id] + 1;
                break;
            default:
                num_frames_ = -1;
                break;
        }
    }

    ~DirectoryIterator() = default;

    pcl::PointCloud<PointT> Next() override {
        int frame_id = frame_id_++;
        pcl::PointCloud<PointT> pc;
        pc = read_pointcloud(options_, sequence_id_, frame_id);
        return pc;
    }

    [[nodiscard]] bool HasNext() const override {
        return frame_id_ < num_frames_;
    }

    void SetInitFrame(int frame_index) override {
        DatasetSequence::SetInitFrame(frame_index);
        frame_id_ = frame_index;
    }

    bool WithRandomAccess() const override { return true; }

    size_t NumFrames() const override { return num_frames_ - init_frame_id_; }

    pcl::PointCloud<PointT> Frame(size_t index) const override {
        int frame_id = index;
        auto pc = read_pointcloud(options_, sequence_id_, frame_id);
        return pc;
    }


private:

    DatasetOptions options_;
    int sequence_id_;
    int frame_id_ = 0;
    int num_frames_;
};

/* -------------------------------------------------------------------------------------------------------------- */
std::shared_ptr<DatasetSequence> get_dataset_sequence(const DatasetOptions &options, int sequence_id) {
    switch (options.dataset) {
        case KITTI_raw:
        case KITTI:
            return std::make_shared<DirectoryIterator>(options, sequence_id);
        default:
            throw std::runtime_error("Not Implemented Error");            
        }
}


}
