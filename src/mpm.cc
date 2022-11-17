#include <memory>

#include "factory.h"
#include "io.h"
#include "mpm.h"
#include "mpm_explicit.h"
#include "mpm_explicit_twophase.h"
#include "mpm_implicit.h"
#include "mpm_semi_implicit_navierstokes.h"
#include "mpm_semi_implicit_twophase.h"
#include "xmpm_explicit.h"

namespace mpm {
// 2D Explicit MPM
static Register<mpm::MPM, mpm::MPMExplicit<2>, const std::shared_ptr<mpm::IO>&>
    mpm_explicit_2d("MPMExplicit2D");

// 3D Explicit MPM
static Register<mpm::MPM, mpm::MPMExplicit<3>, const std::shared_ptr<mpm::IO>&>
    mpm_explicit_3d("MPMExplicit3D");

// 2D Implicit MPM
static Register<mpm::MPM, mpm::MPMImplicit<2>, const std::shared_ptr<mpm::IO>&>
    mpm_implicit_2d("MPMImplicit2D");

// 3D Implicit MPM
static Register<mpm::MPM, mpm::MPMImplicit<3>, const std::shared_ptr<mpm::IO>&>
    mpm_implicit_3d("MPMImplicit3D");

// 2D SemiImplicit Navier Stokes MPM
static Register<mpm::MPM, mpm::MPMSemiImplicitNavierStokes<2>,
                const std::shared_ptr<mpm::IO>&>
    mpm_semi_implicit_navierstokes_2d("MPMSemiImplicitNavierStokes2D");

// 3D SemiImplicit Navier Stokes MPM
static Register<mpm::MPM, mpm::MPMSemiImplicitNavierStokes<3>,
                const std::shared_ptr<mpm::IO>&>
    mpm_semi_implicit_navierstokes_3d("MPMSemiImplicitNavierStokes3D");

// 2D Explicit Two Phase MPM
static Register<mpm::MPM, mpm::MPMExplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    mpm_explicit_twophase_2d("MPMExplicitTwoPhase2D");

// 3D Explicit Two Phase MPM
static Register<mpm::MPM, mpm::MPMExplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    mpm_explicit_twophase_3d("MPMExplicitTwoPhase3D");

// 2D SemiImplicit Two Phase MPM
static Register<mpm::MPM, mpm::MPMSemiImplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    mpm_semi_implicit_twophase_2d("MPMSemiImplicitTwoPhase2D");

// 3D SemiImplicit Two Phase MPM
static Register<mpm::MPM, mpm::MPMSemiImplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    mpm_semi_implicit_twophase_3d("MPMSemiImplicitTwoPhase3D");

// 3D Explicit XMPM
static Register<mpm::MPM, mpm::XMPMExplicit<3>, const std::shared_ptr<mpm::IO>&>
    xmpm_explicit_3d("XMPMExplicit3D");

}  // namespace mpm
