#include "litamin2/litamin2point2voxel.hpp"
#include "litamin2/impl/litamin2point2voxel_impl.hpp"
#include "fast_gicp/gicp/impl/fast_gicp_impl.hpp"

template class litamin::LiTAMIN2Point2Voxel<pcl::PointXYZ, pcl::PointXYZ>;
template class litamin::LiTAMIN2Point2Voxel<pcl::PointXYZI, pcl::PointXYZI>;
