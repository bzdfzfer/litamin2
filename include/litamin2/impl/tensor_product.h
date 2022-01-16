#ifndef TENSOR_PRODUCT_H_
#define TENSOR_PRODUCT_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


inline Eigen::Matrix<double, 9,9> KroneckerProduct3333(const Eigen::Matrix3d& A, 
														const Eigen::Matrix3d& B) {
  Eigen::Matrix<double,9,9> res;
  res.setZero();
  for(int r=0; r < 3; r++)
  	for(int c=0; c < 3; c++) {
  		res.block(r*3, c*3, 3, 3) = A(r, c)*B;
  	}
  return res;
}

inline Eigen::MatrixXd KroneckerProductI44xx(const Eigen::MatrixXd& X) {
	Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
	Eigen::MatrixXd P2(I4.rows() * X.rows(), I4.cols() * X.cols());
	P2.setZero();
	for (int i = 0; i < I4.RowsAtCompileTime; i++)
	{
	    P2.block(i*X.rows(), i*X.cols(), X.rows(), X.cols()) = I4(i, i) * X;
	}  
	return P2;
}


// from Wiki: https://en.wikipedia.org/wiki/Commutation_matrix
// for i = 1 to m
//   for j = 1 to n
//     K(i + m*(j - 1), j + n*(i - 1)) = 1
//   end
// end	
inline Eigen::Matrix<double, 9, 9> getCommutationMatrix33() {
	const int m=3, n=3;
	Eigen::Matrix<double,m*n, m*n> K33;

	K33.setZero();

	for(int i=1; i <= m; i++)
		for(int j=1; j <= n; j++)
			K33(i+m*(j-1)-1, j+n*(i-1)-1) = 1;

	return K33;
}

#endif