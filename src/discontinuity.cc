#include "discontinuity_3d.h"
#include "discontinuity_base.h"
#include "factory.h"

// Triangle 3-noded element
static Register<mpm::DiscontinuityBase<3>, mpm::Discontinuity3D<3>, const Json&>
    tri3d("3d");
