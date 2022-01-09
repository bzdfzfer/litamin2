#include <fast_gicp/gicp/lsq_registration.hpp>

#include <boost/format.hpp>
#include <fast_gicp/so3/so3.hpp>

namespace fast_gicp {

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::LsqRegistration() {
  this->reg_name_ = "LsqRegistration";
  max_iterations_ = 64;
  rotation_epsilon_ = 2e-3;
  transformation_epsilon_ = 5e-4;

  lsq_optimizer_type_ = LSQ_OPTIMIZER_TYPE::LevenbergMarquardt;
  lm_debug_print_ = false;
  lm_max_iterations_ = 10;
  lm_init_lambda_factor_ = 1e-9;
  lm_lambda_ = -1.0;
  lm_sigma1_ = 1e-6;
  lm_sigma2_ = 1e-6;

  final_hessian_.setIdentity();
}

template <typename PointTarget, typename PointSource>
LsqRegistration<PointTarget, PointSource>::~LsqRegistration() {}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setRotationEpsilon(double eps) {
  rotation_epsilon_ = eps;
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setInitialLambdaFactor(double init_lambda_factor) {
  lm_init_lambda_factor_ = init_lambda_factor;
}
template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setLSQType(LSQ_OPTIMIZER_TYPE type) {
  lsq_optimizer_type_ = type;
}
template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::setDebugPrint(bool lm_debug_print) {
  lm_debug_print_ = lm_debug_print;
}

template <typename PointTarget, typename PointSource>
const Eigen::Matrix<double, 6, 6>& LsqRegistration<PointTarget, PointSource>::getFinalHessian() const {
  return final_hessian_;
}

template <typename PointTarget, typename PointSource>
double LsqRegistration<PointTarget, PointSource>::evaluateCost(const Eigen::Matrix4f& relative_pose, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  return this->linearize(Eigen::Isometry3f(relative_pose).cast<double>(), H, b);
}

template <typename PointTarget, typename PointSource>
void LsqRegistration<PointTarget, PointSource>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  Eigen::Isometry3d x0 = Eigen::Isometry3d(guess.template cast<double>());

  lm_lambda_ = -1.0;
  converged_ = false;

  if (lm_debug_print_) {
    std::cout << "********************************************" << std::endl;
    std::cout << "***************** optimize *****************" << std::endl;
    std::cout << "********************************************" << std::endl;
  }

  for (int i = 0; i < max_iterations_ && !converged_; i++) {
    nr_iterations_ = i;

    Eigen::Isometry3d delta;
    if (!step_optimize(x0, delta)) {
      std::cerr << "lm not converged!!" << std::endl;
      break;
    }

    converged_ = is_converged(delta);
  }

  final_transformation_ = x0.cast<float>().matrix();
  pcl::transformPointCloud(*input_, output, final_transformation_);
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::is_converged(const Eigen::Isometry3d& delta) const {
  double accum = 0.0;
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon_ * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_optimize(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  switch (lsq_optimizer_type_) {
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardt:
      return step_lm(x0, delta);
    case LSQ_OPTIMIZER_TYPE::GaussNewton:
      return step_gn(x0, delta);
    case LSQ_OPTIMIZER_TYPE::LevenbergMarquardtNew:
      return step_lm_new(x0, delta);
    case LSQ_OPTIMIZER_TYPE::CeresDogleg:
      return step_ceres(x0,delta);
  }

  return step_lm(x0, delta);
}
template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::solve_ceres(Eigen::Isometry3d& trans,Eigen::Isometry3d& delta) {
  std::cout << "This is lsq_restration method, your code is wrong! You should call child class's method" << std::endl;
  return true;
}
template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_gn(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = linearize(x0, &H, &b);

  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H);
  Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

  delta.setIdentity();
  delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
  delta.translation() = d.tail<3>();

  x0 = delta * x0;
  final_hessian_ = H;

  return true;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = linearize(x0, &H, &b);

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * H.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  for (int i = 0; i < lm_max_iterations_; i++) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    Eigen::Isometry3d xi = delta * x0;
    double yi = compute_error(xi);
    double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b));

    if (lm_debug_print_) {
      if (i == 0) {
        std::cout << boost::format("--- LM optimization ---\n%5s %15s %15s %15s %15s %15s %5s\n") % "i" % "y0" % "yi" % "rho" % "lambda" % "|delta|" % "dec";
      }
      char dec = rho > 0.0 ? 'x' : ' ';
      std::cout << boost::format("%5d %15g %15g %15g %15g %15g %5c") % i % y0 % yi % rho % lm_lambda_ % d.norm() % dec << std::endl;
    }

    if (rho < 0) {
      if (is_converged(delta)) {
        return true;
      }

      lm_lambda_ = nu * lm_lambda_;
      nu = 2 * nu;
      continue;
    }

    x0 = xi;
    lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    final_hessian_ = H;
    return true;
  }

  return false;
}

template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_lm_new(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  Eigen::Matrix<double, 6, 6> H;
  Eigen::Matrix<double, 6, 1> b;
  double y0 = linearize(x0, &H, &b);
  // 若J为0，则说明此处导数为0，且局部最优解=全局最优解
  if (b.array().abs().maxCoeff() <= lm_sigma1_) {
    return true;
  }
  // tau越大，越接近最速；tau越小越接近GN，速度越慢；
  if (lm_lambda_ < 0.0) {
    lm_lambda_ = lm_init_lambda_factor_ * H.diagonal().array().abs().maxCoeff();
  }

  double nu = 2.0;
  for (int i = 0; i < lm_max_iterations_; i++) {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(H + lm_lambda_ * Eigen::Matrix<double, 6, 6>::Identity());
    // d= delta x
    Eigen::Matrix<double, 6, 1> d = solver.solve(-b);

    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();
    // delta x基本不移动了，说明收敛了,与ICP收敛判据冲突，因此进行替换
    // Eigen::Vector3d x_v = so3_log(x0);
    // if(d.norm() <= lm_sigma2_*(x_v.norm()+lm_sigma2_)){
    //   return true;
    // }
    if (is_converged(delta)) {
      return true;
    }
    // xi = x + delta x
    Eigen::Isometry3d xi = delta * x0;
    // yi = F(x+delta x)
    double yi = compute_error(xi);
    double rho = (y0 - yi) / (0.5 * d.dot(lm_lambda_ * d - b));

    if (rho < 0) {
      lm_lambda_ = nu * lm_lambda_;
      nu = 2 * nu;
      continue;
    }

    x0 = xi;
    lm_lambda_ = lm_lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    final_hessian_ = H;
    return true;
  }

  return false;
}
template <typename PointTarget, typename PointSource>
bool LsqRegistration<PointTarget, PointSource>::step_ceres(Eigen::Isometry3d& x0, Eigen::Isometry3d& delta) {
  return solve_ceres(x0,delta);
}
}  // namespace fast_gicp