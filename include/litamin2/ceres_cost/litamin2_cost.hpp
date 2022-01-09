#ifndef REGISTRATION_COST_HPP
#define REGISTRATION_COST_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

struct LiTAMIN2CostFunction
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LiTAMIN2CostFunction(Eigen::Vector3d p_mean, Eigen::Vector3d q_mean, Eigen::Matrix3d p_cov, Eigen::Matrix3d q_cov,
                           Eigen::Matrix3d lambdaI,double sigma_ICP) :
                        p_mean_(p_mean), q_mean_(q_mean), p_cov_(p_cov), q_cov_(q_cov),lambdaI_(lambdaI),sigma_ICP_(sigma_ICP) {}

    template <typename T>
    bool operator()(const T *const q, const T *const t, T *residuals) const
    {
        Eigen::Map<Eigen::Matrix<T,1,1>> residuals_map(residuals);
        Eigen::Matrix<T, 3, 1> p_m(p_mean_.cast<T>());
        Eigen::Matrix<T, 3, 1> q_m(q_mean_.cast<T>());
        Eigen::Matrix<T, 3, 3> p_c = p_cov_.cast<T>();
        Eigen::Matrix<T, 3, 3> q_c = q_cov_.cast<T>();
        Eigen::Matrix<T, 3, 3> lambI = lambdaI_.cast<T>();       
        Eigen::Quaternion<T> quat(q);
        Eigen::Matrix<T, 3, 1> translation(t);

        Eigen::Matrix<T, 3, 3> mahalanobis = (q_c + quat * p_c * (quat.inverse()) + lambdaI_).inverse();
        mahalanobis.normalize();
        Eigen::Matrix<T,3,3> LT = mahalanobis.llt().matrixL().transpose();
        Eigen::Matrix<T,3,1> residuals_err = LT * (q_m - (quat * p_m + translation));
        T EICP = T(residuals_err.squaredNorm());
        T sigma_square = T(sigma_ICP_*sigma_ICP_);
        // residuals_map = (T(1.)-(EICP/(EICP+sigma_square)))*residuals_err;
        T wICP = T(1.)-EICP/(EICP+sigma_square);
        residuals[0] = wICP*EICP;

        // T sigma_cov_square = T(3.0*3.0);
        // Eigen::Matrix<T,3,3> R = quat.toRotationMatrix();
        // Eigen::Matrix<T,3,3> RT = R.transpose();
        // Eigen::Matrix<T,3,3> q_cRp_icRT = q_c * R * p_c.inverse() * RT;
        // Eigen::Matrix<T,3,3> q_icRp_cRT = q_c.inverse() * R * p_c * RT;
        // T ECOV = q_cRp_icRT.trace() + q_cRp_icRT.trace() - T(6);
        // T wCov = T(1.)-ECOV/(ECOV+sigma_cov_square);
        // residuals[0] += wCov*ECOV;

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d p_mean_, Eigen::Vector3d q_mean_, Eigen::Matrix3d p_cov_, 
    Eigen::Matrix3d q_cov_,Eigen::Matrix3d lambdaI_, double sigma_ICP_){
        // 残差是三维的,变量分别是四维和三维
        return (new ceres::AutoDiffCostFunction<LiTAMIN2CostFunction,1,4,3>
        (new LiTAMIN2CostFunction(p_mean_,q_mean_,p_cov_,q_cov_,lambdaI_,sigma_ICP_)));
    }
    
    Eigen::Vector3d p_mean_, q_mean_;
    Eigen::Matrix3d p_cov_, q_cov_;
    Eigen::Matrix3d lambdaI_;
    double sigma_ICP_;
};
#endif