// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _LIDAR_OPTIMIZATION_ANALYTIC_H_
#define _LIDAR_OPTIMIZATION_ANALYTIC_H_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

Eigen::Matrix3d skew(Eigen::Vector3d& mat_in) {
  Eigen::Matrix<double, 3, 3> skew_mat;
  skew_mat.setZero();
  skew_mat(0, 1) = -mat_in(2);
  skew_mat(0, 2) = mat_in(1);
  skew_mat(1, 2) = -mat_in(0);
  skew_mat(1, 0) = mat_in(2);
  skew_mat(2, 0) = -mat_in(1);
  skew_mat(2, 1) = mat_in(0);
  return skew_mat;
}

void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t) {
  Eigen::Vector3d omega(se3.data());
  Eigen::Vector3d upsilon(se3.data() + 3);
  Eigen::Matrix3d Omega = skew(omega);

  double theta = omega.norm();
  double half_theta = 0.5 * theta;

  double imag_factor;
  double real_factor = cos(half_theta);
  if (theta < 1e-10) {
    double theta_sq = theta * theta;
    double theta_po4 = theta_sq * theta_sq;
    imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
  } else {
    double sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta / theta;
  }

  q = Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());

  Eigen::Matrix3d J;
  if (theta < 1e-10) {
    J = q.matrix();
  } else {
    Eigen::Matrix3d Omega2 = Omega * Omega;
    J = (Eigen::Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * Omega + (theta - sin(theta)) / (pow(theta, 3)) * Omega2);
  }

  t = J * upsilon;
}



class PoseSE3Parameterization : public ceres::LocalParameterization {
public:
  PoseSE3Parameterization() {}
  virtual ~PoseSE3Parameterization() {}
  // 从这里看，参数块应该是[q,t]，且q是[x,y,z,w]
  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);

    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_t;
    getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta), delta_q, delta_t);
    Eigen::Map<const Eigen::Quaterniond> quater(x);
    Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);
    // TODO 不需要归一化吗？
    quater_plus = delta_q * quater;
    trans_plus = delta_q * trans + delta_t;

    return true;
  }
  // ref: https://github.com/wh200720041/floam/issues/50
  // 这是全局参数到局部参数的雅可比矩阵。全局是trans,quat，因此是7维，局部则是se3,因此是6维。
  // 答案中提到了J1*J2,其中J1是参数块对[t,q]求导来自于evaluate或者AD，J2则是这里的到李代数的参数化。
  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    (j.topRows(6)).setIdentity();
    (j.bottomRows(1)).setZero();

    return true;
  }
  virtual int GlobalSize() const { return 7; }
  virtual int LocalSize() const { return 6; }
};

#endif  // _LIDAR_OPTIMIZATION_ANALYTIC_H_
