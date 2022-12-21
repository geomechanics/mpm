#include "factory.h"
#include "point_base.h"
#include "point_penalty_displacement.h"

// PointPenaltyDisplacement2D (2 Dim)
static Register<mpm::PointBase<2>, mpm::PointPenaltyDisplacement<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    point2dpenaltydisplacement("POINT2DPENALTYDISP");

// PointPenaltyDisplacement3D (3 Dim)
static Register<mpm::PointBase<3>, mpm::PointPenaltyDisplacement<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    point3dpenaltydisplacement("POINT3DPENALTYDISP");