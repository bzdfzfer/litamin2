#include "litamin2/litamin2.hpp"
#include "litamin2/impl/litamin2_impl.hpp"
#include "fast_gicp/gicp/impl/fast_gicp_impl.hpp"

template class litamin::LiTAMIN2<pcl::PointXYZ, pcl::PointXYZ>;
template class litamin::LiTAMIN2<pcl::PointXYZI, pcl::PointXYZI>;
