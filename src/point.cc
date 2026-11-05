#include "factory.h"
#include "point_base.h"
#include "point_dirichlet_direct.h"
#include "point_dirichlet_penalty.h"

namespace mpm {
// PointType
std::map<std::string, int> PointType = {{"POINT2DDIRDIRECT", 0},
                                        {"POINT3DDIRDIRECT", 1},
                                        {"POINT2DDIRPEN", 2},
                                        {"POINT3DDIRPEN", 3}};
std::map<int, std::string> PointTypeName = {{0, "POINT2DDIRDIRECT"},
                                            {1, "POINT3DDIRDIRECT"},
                                            {2, "POINT2DDIRPEN"},
                                            {3, "POINT3DDIRPEN"}};
std::map<std::string, std::string> PointPODTypeName = {
    {"POINT2DDIRDIRECT", "points"},
    {"POINT3DDIRDIRECT", "points"},
    {"POINT2DDIRPEN", "points"},
    {"POINT3DDIRPEN", "points"}};
}  // namespace mpm

// PointDirichletDirect2D (2 Dim)
static Register<mpm::PointBase<2>, mpm::PointDirichletDirect<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    point2ddirichletdirect("POINT2DDIRDIRECT");

// PointDirichletDirect3D (3 Dim)
static Register<mpm::PointBase<3>, mpm::PointDirichletDirect<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    point3ddirichletdirect("POINT3DDIRDIRECT");

// PointDirichletPenalty2D (2 Dim)
static Register<mpm::PointBase<2>, mpm::PointDirichletPenalty<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    point2ddirichletpenalty("POINT2DDIRPEN");

// PointDirichletPenalty3D (3 Dim)
static Register<mpm::PointBase<3>, mpm::PointDirichletPenalty<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    point3ddirichletpenalty("POINT3DDIRPEN");