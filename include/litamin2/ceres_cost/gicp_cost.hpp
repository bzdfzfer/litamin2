#ifndef GICP_COST_HPP
#define GICP_COST_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "litamin2/ceres_cost/PoseSE3Parameterization.hpp"
// L^T*error
struct GICP_FACTOR {
  GICP_FACTOR(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov) : p_mean_(p_mean), q_mean_(q_mean), p_cov_(p_cov), q_cov_(q_cov) {}

  template <typename T>
  bool operator()(const T* const q, const T* const t, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals);
    Eigen::Matrix<T, 3, 1> p_m(p_mean_.cast<T>());
    Eigen::Matrix<T, 3, 1> q_m(q_mean_.cast<T>());
    Eigen::Matrix<T, 3, 3> p_c = p_cov_.cast<T>();
    Eigen::Matrix<T, 3, 3> q_c = q_cov_.cast<T>();

    Eigen::Quaternion<T> quat(q);
    Eigen::Matrix<T, 3, 1> translation(t);

    Eigen::Matrix<T, 3, 3> mahalanobis = (q_c + quat * p_c * quat.inverse()).inverse();
    Eigen::Matrix<T, 3, 3> LT = mahalanobis.llt().matrixL().transpose();
    residuals_map = LT * (q_m - (quat * p_m + translation));

    return true;
  }

  static ceres::CostFunction* Create(Eigen::Vector3d p_mean_, Eigen::Vector3d q_mean_, Eigen::Matrix3d p_cov_, Eigen::Matrix3d q_cov_) {
    // 分别是残差，q，t的维度
    return (new ceres::AutoDiffCostFunction<GICP_FACTOR, 3, 4, 3>(new GICP_FACTOR(p_mean_, q_mean_, p_cov_, q_cov_)));
  }

  Eigen::Vector3d p_mean_, q_mean_;
  Eigen::Matrix3d p_cov_, q_cov_;
};

class GICPAnalyticCostFunction : public ceres::SizedCostFunction<3, 7> {
public:
  GICPAnalyticCostFunction(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov)
  : p_mean_(p_mean),
    q_mean_(q_mean),
    p_cov_(p_cov),
    q_cov_(q_cov) {}
  virtual ~GICPAnalyticCostFunction() {}
  // parameters是[0,0,0,1,x,y,z]
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    // 默认已经归一化
    Eigen::Quaterniond q_last_curr(parameters[0]);
    Eigen::Vector3d t_last_curr(parameters[0] + 4);
    Eigen::Matrix3d LT = (q_cov_ + q_last_curr * q_cov_ * q_last_curr.inverse()).inverse().llt().matrixL().transpose();
    Eigen::Map<Eigen::Vector3d> residuals_map(residuals);
    Eigen::Vector3d p_mean_trans = q_last_curr * p_mean_ + t_last_curr;
    residuals_map = LT * (q_mean_ - p_mean_trans);

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
        J_se3.setZero();
        Eigen::Matrix<double, 3, 6> dp_by_se3;
        dp_by_se3.block<3, 3>(0, 0) = skew(p_mean_trans);
        dp_by_se3.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
        J_se3.block<3, 6>(0, 0) = LT * dp_by_se3;
      }
    }
    return true;
  }

  Eigen::Vector3d p_mean_, q_mean_;
  Eigen::Matrix3d p_cov_, q_cov_;
};

// 弃用
// 精度不高，而且耗时
// Z是R*Sigma_i*RT对李代数的雅可比矩阵，来自于d2d-ndt，具体如何计算，我不清楚
class GICPDoubleAnalyticCostFunction : public ceres::SizedCostFunction<3, 7> {
public:
  GICPDoubleAnalyticCostFunction(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov)
  : p_mean_(p_mean),
    q_mean_(q_mean),
    p_cov_(p_cov),
    q_cov_(q_cov) {}
  virtual ~GICPDoubleAnalyticCostFunction() {}
  // parameters是[0,0,0,1,x,y,z]
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    // 默认已经归一化
    Eigen::Quaterniond q_last_curr(parameters[0]);
    Eigen::Vector3d t_last_curr(parameters[0] + 4);
    Eigen::Matrix3d mahalanobis = (q_cov_ + q_last_curr * q_cov_ * q_last_curr.inverse()).inverse();
    // Eigen::Matrix3d LT = mahalanobis.llt().matrixL().transpose();
    Eigen::Map<Eigen::Vector3d> residuals_map(residuals);
    Eigen::Vector3d p_mean_trans = q_last_curr * p_mean_ + t_last_curr;
    residuals_map = mahalanobis * (q_mean_ - p_mean_trans);

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
        J_se3.setZero();
        Eigen::Matrix<double, 3, 6> dp_by_se3;
        Eigen::Matrix<double, 3, 6> BZBU(Eigen::Matrix<double,3,6>::Zero());
        dp_by_se3.block<3, 3>(0, 0) = skew(p_mean_trans);
        dp_by_se3.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
        Eigen::Matrix3d Z1, Z2, Z3;
        Z1 << 0, -p_cov_(0, 2), p_cov_(0, 1), -p_cov_(0, 2), -2 * p_cov_(1, 2), -p_cov_(2, 2) + p_cov_(1, 1), p_cov_(0, 1), -p_cov_(2, 2) + p_cov_(1, 1), 2 * p_cov_(1, 2);
        Z2 << 2 * p_cov_(0, 2), p_cov_(1, 2), -p_cov_(0, 0) + p_cov_(2, 2), p_cov_(1, 2), 0, -p_cov_(0, 1), -p_cov_(0, 0) + p_cov_(2, 2), -p_cov_(0, 1), -2 * p_cov_(0, 2);
        Z3 << -2 * p_cov_(0, 1), -p_cov_(1, 1) + p_cov_(0, 0), -p_cov_(1, 2), -p_cov_(1, 1) + p_cov_(0, 0), 2 * p_cov_(0, 1), p_cov_(0, 2), -p_cov_(1, 2), p_cov_(0, 2), 0;
        BZBU.block<3,1>(0,0) = mahalanobis * Z1 * mahalanobis * p_mean_trans;
        BZBU.block<3,1>(0,1) = mahalanobis * Z2 * mahalanobis * p_mean_trans;
        BZBU.block<3,1>(0,2) = mahalanobis * Z3 * mahalanobis * p_mean_trans;
        J_se3.block<3, 6>(0, 0) = -BZBU + mahalanobis * dp_by_se3;
      }
    }
    return true;
  }

  Eigen::Vector3d p_mean_, q_mean_;
  Eigen::Matrix3d p_cov_, q_cov_;
};

// 正常工作，但精度并不是特别的高
// TODO 删除无效代码
class ICPAnalyticCostFunction : public ceres::SizedCostFunction<3, 7> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ICPAnalyticCostFunction(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov)
  : p_mean_(p_mean),
    q_mean_(q_mean),
    p_cov_(p_cov),
    q_cov_(q_cov) {}
  virtual ~ICPAnalyticCostFunction() {}
  // parameters是[0,0,0,1,x,y,z]
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
    // 默认已经归一化
    Eigen::Quaterniond q_last_curr(parameters[0]);
    Eigen::Vector3d t_last_curr(parameters[0] + 4);
    // Eigen::Matrix3d mahalanobis = q_cov_ + q_last_curr * q_cov_ * q_last_curr.inverse();
    // Eigen::Matrix3d LT = mahalanobis.llt().matrixL().transpose();
    Eigen::Map<Eigen::Vector3d> residuals_map(residuals);
    Eigen::Vector3d p_mean_trans = q_last_curr * p_mean_ + t_last_curr;
    residuals_map = q_mean_ - p_mean_trans;

    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
        J_se3.setZero();
        Eigen::Matrix<double, 3, 6> dp_by_se3;
        dp_by_se3.block<3, 3>(0, 0) = skew(p_mean_trans);
        dp_by_se3.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
        J_se3.block<3, 6>(0, 0) = dp_by_se3;
      }
    }
    return true;
  }

  Eigen::Vector3d p_mean_, q_mean_;
  Eigen::Matrix3d p_cov_, q_cov_;
};
#endif