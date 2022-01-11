#ifndef LITAMIN2POINT2VOXEL_IMPL_HPP
#define LITAMIN2POINT2VOXEL_IMPL_HPP 

#include <atomic>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "litamin2/ceres_cost/litamin2_cost.hpp"

#include "fast_gicp/so3/so3.hpp"
#include "litamin2/litamin2point2voxel.hpp"

#include "fast_gicp/time_utils.hpp"


using namespace fast_gicp;

namespace litamin {


template <typename PointSource, typename PointTarget>
LiTAMIN2Point2Voxel<PointSource, PointTarget>::LiTAMIN2Point2Voxel() : FastGICP<PointSource, PointTarget>() {
  this->reg_name_ = "LiTAMIN2Point2Voxel";

  voxel_resolution_ = 0.5; // default voxel resolution is 3m by 3m by 3m.
  search_method_ = NeighborSearchMethod::DIRECT1;
  voxel_mode_ = VoxelAccumulationMode::ADDITIVE;
  this->setRegularizationMethod(RegularizationMethod::NONE);
}

template <typename PointSource, typename PointTarget>
LiTAMIN2Point2Voxel<PointSource, PointTarget>::~LiTAMIN2Point2Voxel() {}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::setResolution(double resolution) {
  voxel_resolution_ = resolution;
}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::setNeighborSearchMethod(NeighborSearchMethod method) {
  search_method_ = method;
}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::setVoxelAccumulationMode(VoxelAccumulationMode mode) {
  voxel_mode_ = mode;
}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);
  target_voxelmap_.reset();
  voxel_correspondences_.clear();
  voxel_mahalanobis_.clear();
}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  // if (target_ == cloud) {
  //   return;
  // }

  FastGICP<PointSource, PointTarget>::setInputTarget(cloud);
  target_voxelmap_.reset();
}

template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  target_voxelmap_.reset();

  FastGICP<PointSource, PointTarget>::computeTransformation(output, guess);
}




/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// key implementation ////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


template <typename PointSource, typename PointTarget>
void LiTAMIN2Point2Voxel<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  voxel_correspondences_.clear();
  auto offsets = neighbor_offsets(search_method_);

  // std::cout << "input_ voxel size: " << input_->size() << std::endl;

  std::vector<std::vector<std::pair<int, GaussianVoxel::Ptr>>> corrs(num_threads_);
  for (auto& c : corrs) {
    c.reserve((input_->size() * offsets.size()) / num_threads_);
  }

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    Eigen::Vector4d transed_mean_A = trans * mean_A;
    Eigen::Vector3i coord = target_voxelmap_->voxel_coord(transed_mean_A);

    for (const auto& offset : offsets) {
      auto voxel = target_voxelmap_->lookup_voxel(coord + offset);
      if (voxel != nullptr) {
        int thread_num = omp_get_thread_num();
        corrs[thread_num].push_back(std::make_pair(i, voxel));
      }
    }
  }

  voxel_correspondences_.reserve(input_->size() * offsets.size() );

  for (const auto& c : corrs) {
    voxel_correspondences_.insert(voxel_correspondences_.end(), c.begin(), c.end());
  }


  // std::cout << "voxel_correspondences_ size: " << voxel_correspondences_.size() << std::endl;
  // precompute combined covariances
  voxel_mahalanobis_.resize(voxel_correspondences_.size());

  // calculate covariances: Cqp = Cq + R*Cp*R.transpose() + lambda*I
  // q is target, p is input. here a is input, b is target.
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    const auto& cov_A = source_covs_[corr.first];
    const auto& cov_B = corr.second->cov;
    
    
    // original implementation.
    // Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    // RCR(3, 3) = 1.0;
    // voxel_mahalanobis_[i] = RCR.inverse();
    // voxel_mahalanobis_[i](3, 3) = 0.0;
    
    double lambda = 1e-6;
    Eigen::Matrix4d RCART = trans.matrix() * cov_A * trans.matrix().transpose();
    Eigen::Matrix3d C = cov_B.block<3, 3>(0, 0).cast<double>() + RCART.block<3,3>(0,0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d C_inv = C.inverse();
    voxel_mahalanobis_[i].setZero();
    voxel_mahalanobis_[i].template block<3, 3>(0, 0) = C_inv / C_inv.norm();
  }

}

template <typename PointSource, typename PointTarget>
double LiTAMIN2Point2Voxel<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  if (target_voxelmap_ == nullptr) {
    target_voxelmap_.reset(new GaussianVoxelMap<PointTarget>(voxel_resolution_, voxel_mode_));
    target_voxelmap_->create_voxelmap(*target_, target_covs_);
    // std::cout << "target_voxelmap_ size: " << target_voxelmap_->voxel_size() << std::endl;
  }

  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }
    
    // TODO
    // calculate covariance cost function, 
    // Ecov = Tr(R*Cp.inv()*R.transpose()*Cq) + Tr(Cq.inv()*R*Cp*R.transpose())-6
#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const auto& cov_B = corr.second->cov;

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    // implementation only consider ICP cost.
    double w = std::sqrt(target_voxel->num_points);
    w = 1;
    // sum_errors += w * error.transpose() * voxel_mahalanobis_[i] * error;


    // new implementation considering ICP cost and covariance cost.    
    const double wICPThreshold = 0.5;
    const double wCovThreshold = 3;
    double w_icp, w_cov;

    double icp_error = error.transpose() * voxel_mahalanobis_[i] * error;
    w_icp = 1 - icp_error / (icp_error + wICPThreshold);
    w = w_icp;
    sum_errors += w_icp * icp_error;

    // construct Covariance shape error.
    // Eigen::Matrix4d CbRCaiRT = cov_B * trans.matrix() * cov_A.inverse() * trans.matrix().transpose();
    // Eigen::Matrix4d CbiRCaRT = cov_B.inverse() * trans.matrix() * cov_A * trans.matrix().transpose();
    // double cov_error = CbRCaiRT.block<3,3>(0,0).trace() + CbiRCaRT.block<3,3>(0,0).trace() - 6;
    // w_cov = 1 - cov_error / (cov_error + wCovThreshold);

    // sum_errors += w_cov * cov_error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = w * jlossexp.transpose() * voxel_mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = w * jlossexp.transpose() * voxel_mahalanobis_[i] * error;

    int thread_num = omp_get_thread_num();
    Hs[thread_num] += Hi;
    bs[thread_num] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double LiTAMIN2Point2Voxel<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;
#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors)
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const auto& cov_B = corr.second->cov;

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    // double w = std::sqrt(target_voxel->num_points);
    // w = 1;
    // sum_errors += w * error.transpose() * voxel_mahalanobis_[i] * error;

    // new implementation considering ICP cost and covariance cost.    
    const double sigmaICP = 0.5;
    const double sigmaCov = 3;
    double w_icp, w_cov;

    double icp_error = error.transpose() * voxel_mahalanobis_[i] * error;
    w_icp = 1 - icp_error / (icp_error + sigmaICP*sigmaICP);
    sum_errors += w_icp * icp_error;    

    // Eigen::Matrix4d CbRCaiRT = cov_B * trans.matrix() * cov_A.inverse() * trans.matrix().transpose();
    // Eigen::Matrix4d CbiRCaRT = cov_B.inverse() * trans.matrix() * cov_A * trans.matrix().transpose();
    // double cov_error = CbRCaiRT.block<3,3>(0,0).trace() + CbiRCaRT.block<3,3>(0,0).trace() - 6;
    // w_cov = 1 - cov_error / (cov_error + sigmaCov);
    // sum_errors += w_cov * cov_error;

  }

  return sum_errors;
}

// 1. 更新对应关系
// 2. 构建优化方程
// 3. 求解
// 4. 改变trans,输出delta
template <typename PointSource, typename PointTarget>
bool LiTAMIN2Point2Voxel<PointSource, PointTarget>::solve_ceres(Eigen::Isometry3d& trans, Eigen::Isometry3d& delta) {
  if (target_voxelmap_ == nullptr) {
    target_voxelmap_.reset(new GaussianVoxelMap<PointTarget>(voxel_resolution_, voxel_mode_));
    target_voxelmap_->create_voxelmap(*target_, target_covs_);
    // std::cout << "target_voxelmap_ size: " << target_voxelmap_->voxel_size() << std::endl;    
  }  
  tic::TicToc t;
  update_correspondences(trans);
  // std::cout << "update correspondences time: " << t.toc() << " ms. " << std::endl;
  Eigen::Isometry3d origin = trans;
  Eigen::Quaterniond q_guess(origin.linear());
  Eigen::Vector3d t_guess(origin.translation());
  // Eigen四元数构造是wxyz，存储是xyzw，本库全部采用eigen表达方法
  double para[7] = {q_guess.x(), q_guess.y(), q_guess.z(), q_guess.w(),t_guess[0], t_guess[1], t_guess[2]};
  double* para_q;
  double* para_t;
  para_q = &para[0];
  para_t = &para[4];
  ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  problem.AddParameterBlock(para_q, 4, q_parameterization);
  problem.AddParameterBlock(para_t, 3);


  // q_last是优化向量的映射
  Eigen::Map<Eigen::Quaterniond> q_last(para_q);
  Eigen::Map<Eigen::Vector3d> t_last(para_t);
  t.toc();
  for (int i = 0; i < voxel_correspondences_.size(); i++) {
    const auto& corr = voxel_correspondences_[i];
    auto target_voxel = corr.second;

    const Eigen::Vector4d mean_A = input_->at(corr.first).getVector4fMap().template cast<double>();
    const Eigen::Matrix4d cov_A = source_covs_[corr.first];

    const Eigen::Vector4d mean_B = corr.second->mean;
    const Eigen::Matrix4d cov_B = corr.second->cov;

    const Eigen::Vector3d p_mean = mean_A.block<3,1>(0,0);
    const auto& p_cov = cov_A.block<3,3>(0,0);
    const Eigen::Vector3d q_mean = mean_B.block<3,1>(0,0);
    const auto& q_cov = cov_B.block<3,3>(0,0);

    // const Eigen::Vector3d p_mean = input_->at(i).getVector3fMap().template cast<double>();
    // const auto& p_cov = source_covs_[i].block<3, 3>(0, 0);

    // const Eigen::Vector3d q_mean = target_->at(target_index).getVector3fMap().template cast<double>();
    // const auto& q_cov = target_covs_[target_index].block<3, 3>(0, 0);

    Eigen::Matrix3d lambdaI = 1e-6*Eigen::Matrix3d::Identity();
    double sigmaICP = 0.5;
    ceres::CostFunction* cost_function = LiTAMIN2CostFunction::Create(p_mean, q_mean, p_cov, q_cov, lambdaI, sigmaICP);
    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);


  }
  // cout << "ceres add residual block time: " << t.toc() << " ms. " << std::endl;
  // 开始求解ceres问题
  ceres::Solver::Options options;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  // options.dogleg_type = ceres::TRADITIONAL_DOGLEG;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 10;
  options.function_tolerance = 1e-3;
  options.gradient_tolerance = 1e-6;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = num_threads_;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);  // 求解完毕后，q_last和t_last代表了target->source
  // cout << "solve time: " << t.toc()  << " ms. "  << endl;
  trans.linear() = q_last.matrix();
  trans.translation() = t_last;
  delta = origin.inverse() * trans;
  // cout << "result: " << trans.matrix() << endl;
  return summary.IsSolutionUsable();
}


}

#endif