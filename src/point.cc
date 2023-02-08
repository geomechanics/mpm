#include "factory.h"
#include "point_base.h"
#include "point_dirichlet_penalty.h"

namespace mpm {
// PointType
std::map<std::string, int> PointType = {{"POINT2DDIRPEN", 0},
                                        {"POINT3DDIRPEN", 1}};
std::map<int, std::string> PointTypeName = {{0, "POINT2DDIRPEN"},
                                            {1, "POINT3DDIRPEN"}};
std::map<std::string, std::string> PointPODTypeName = {
    {"POINT2DDIRPEN", "points"}, {"POINT3DDIRPEN", "points"}};
}  // namespace mpm

// PointDirichletPenalty2D (2 Dim)
static Register<mpm::PointBase<2>, mpm::PointDirichletPenalty<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    point2ddirichletpenalty("POINT2DDIRPEN");

// PointDirichletPenalty3D (3 Dim)
static Register<mpm::PointBase<3>, mpm::PointDirichletPenalty<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    point3ddirichletpenalty("POINT2DDIRPEN");