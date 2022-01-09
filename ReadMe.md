# LITAMIN2
This package is an implementation of litamin2 point cloud matching algorithm.


## Dependencies.
ROS
PCL 
CERES
Eigen3

## Build.
``` 
makedir -p litamin_ws/src/ && cd litamin_ws/src/
git clone https://github.com/bzdfzfer/litamin2
cd ~/litamin_ws/ && catkin build

```

## Run test program.

```
source devel/setup.bash
rosrun litamin2 litamin2_align ~/litamin_ws/src/litamin2/data/251370668.pcd ~/litamin_ws/src/litamin2/data/251371071.pcd
```

## Results.
Voxel resolution is set to 0.5m.
[aligned results](data/litamin2_results.png).
- [GT] Two scan transformation matrix:
```
    0.999941    0.0108432 -0.000635437     0.485657
  -0.0108468     0.999924  -0.00587782      0.10642
 0.000571654   0.00588436     0.999983   -0.0131581
           0            0            0            1
```
- LM based optimization with zero lambda.
** speed: repeative run of 100 times, average 33ms.
** accuracy: 
``` bash
align result: 
   0.999867   0.0162508 -0.00157465    0.491412
 -0.0162532    0.999867 -0.00147708    0.117267
 0.00155044  0.00150248    0.999998  -0.0306296
          0           0           0           1
```
- Ceres based solution.
** speed: repeative run of 100 times, average 132ms.
** accuracy: 
``` bash
align result: 
   0.999875   0.0157213 -0.00137114    0.465903
 -0.0157243    0.999874 -0.00218937    0.114884
 0.00133655  0.00221065    0.999997  -0.0321361
          0           0           0           1
```

## References.
* Yokozuka M, Koide K, Oishi S, et al. LiTAMIN2: Ultra Light LiDAR-based SLAM using Geometric Approximation applied with KL-Divergence, ICRA2021. [litamin2 paper link](https://arxiv.org/abs/2103.00784).
* Kenji Koide, Masashi Yokozuka, Shuji Oishi, and Atsuhiko Banno, Voxelized GICP for fast and accurate 3D point cloud registration, ICRA2021. [fast-gicp paper link](https://staff.aist.go.jp/shuji.oishi/assets/papers/preprint/VoxelGICP_ICRA2021.pdf).
* Official implementation of Fast-GICP. [fast-gicp implementation](https://github.com/SMRT-AIST/fast_gicp).
* Another implementation of Fast-GICP using Ceres. [fastgicp re-implementation](https://github.com/FishInWave/fast-gicp).

## TODO.
* Current implementation only uses ICP cost, the covariance rotation cost function is not successfully added. (due to instable convex optimization of trace terms. )
* Find Hessian matrix of E_{ICP} and E_{COV} and implement Newton method mentioned in the paper.
