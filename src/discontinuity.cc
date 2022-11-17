#include "discontinuity_3d.h"
#include "discontinuity_base.h"
#include "factory.h"

// Triangle 3-noded element
static Register<mpm::DiscontinuityBase<3>, mpm::Discontinuity3D<3>, const Json&,
                unsigned>
    tri3d("3d");
static Register<
    mpm::DiscontinuityBase<3>, mpm::Discontinuity3D<3>, unsigned,
    std::tuple<double, double, double, double, double, int, bool, bool>&>
    tri3d_initiation("3d_initiation");