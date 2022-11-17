#include "discontinuity_point.h"
#include "factory.h"
#include "point_base.h"
// Discontinuity Point
static Register<mpm::PointBase<3>, mpm::DiscontinuityPoint<3>,
                const Eigen::Matrix<double, 3, 1>&, mpm::Index>
    discontinuitypoint3d("DiscontinuityPoint3D");
