#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "litamin2/ceres_cost/gicp_cost.hpp"
#include "fast_gicp/time_utils.hpp"
namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  reg_name_ = "FastGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();
  isSE3_ = false;

  regularization_method_ = RegularizationMethod::PLANE;
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointTarget>);
}

template <typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::~FastGICP() {}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  // 移除功能：根据指针来跳过setInput操作
  // if (input_ == cloud) {
  //   return;
  // }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_kdtree_->setInputCloud(cloud);
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  // 移除功能：根据指针来跳过setInput操作
  // if (target_ == cloud) {
  //   return;
  // }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_kdtree_->setInputCloud(cloud);
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (source_covs_.size() != input_->size()) {
    calculate_covariances(input_, *source_kdtree_, source_covs_);
  }
  if (target_covs_.size() != target_->size()) {
    calculate_covariances(target_, *target_kdtree_, target_covs_);
  }

  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());

  Eigen::Isometry3f trans_f = trans.cast<float>();

  correspondences_.resize(input_->size());
  sq_distances_.resize(input_->size());
  mahalanobis_.resize(input_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    sq_distances_[i] = k_sq_dists[0];
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;

    if (correspondences_[i] < 0) {
      continue;
    }

    const int target_index = correspondences_[i];
    const auto& cov_A = source_covs_[i];
    const auto& cov_B = target_covs_[target_index];

    Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
    RCR(3, 3) = 1.0;

    mahalanobis_[i] = RCR.inverse();
    mahalanobis_[i](3, 3) = 0.0f;
  }
}
// 1. 更新对应关系
// 2. 构建优化方程
// 3. 求解
// 4. 改变trans,输出delta
template <typename PointSource, typename PointTarget>
bool FastGICP<PointSource, PointTarget>::solve_ceres(Eigen::Isometry3d& trans, Eigen::Isometry3d& delta) {
  tic::TicToc t;
  update_correspondences(trans);
  std::cout << "update correspondences time: " << t.toc() << std::endl;
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
  if(!isSE3_){
    problem.AddParameterBlock(para_q, 4, q_parameterization);
    problem.AddParameterBlock(para_t, 3);
  }else{
    problem.AddParameterBlock(para, 7 ,new PoseSE3Parameterization());
  }

  // q_last是优化向量的映射
  Eigen::Map<Eigen::Quaterniond> q_last(para_q);
  Eigen::Map<Eigen::Vector3d> t_last(para_t);
  t.toc();
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector3d p_mean = input_->at(i).getVector3fMap().template cast<double>();
    const auto& p_cov = source_covs_[i].block<3, 3>(0, 0);

    const Eigen::Vector3d q_mean = target_->at(target_index).getVector3fMap().template cast<double>();
    const auto& q_cov = target_covs_[target_index].block<3, 3>(0, 0);
    // p是source,q是target
    if(!isSE3_){
      ceres::CostFunction* cost_function = GICP_FACTOR::Create(p_mean, q_mean, p_cov, q_cov);
      problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
    }else{
      ceres::CostFunction* cost_function = new GICPAnalyticCostFunction(p_mean, q_mean, p_cov, q_cov);
      problem.AddResidualBlock(cost_function,loss_function,para);
    }

  }
  cout << "ceres add residual block time: " << t.toc() << std::endl;
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
  cout << "solve time: " << t.toc() << endl;
  trans.linear() = q_last.matrix();
  trans.translation() = t_last;
  delta = origin.inverse() * trans;
  cout << "result: " << trans.matrix() << endl;
  return summary.IsSolutionUsable();
}

template <typename PointSource, typename PointTarget>
double FastGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
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
double FastGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::KdTree<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);

    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;

      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
  }

  return true;
}

}  // namespace fast_gicp

#endif
