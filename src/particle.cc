#include "particle.h"
#include "factory.h"
#include "particle_base.h"
#include "particle_bbar.h"
#include "particle_finite_strain.h"
#include "particle_fluid.h"
#include "particle_twophase.h"

namespace mpm {
// ParticleType
std::map<std::string, int> ParticleType = {
    {"P2D", 0},       {"P3D", 1},       {"P2DFLUID", 2}, {"P3DFLUID", 3},
    {"P2D2PHASE", 4}, {"P3D2PHASE", 5}, {"P2DBBAR", 6},  {"P3DBBAR", 7},
    {"P2DFS", 8},     {"P3DFS", 9}};
std::map<int, std::string> ParticleTypeName = {
    {0, "P2D"},       {1, "P3D"},       {2, "P2DFLUID"}, {3, "P3DFLUID"},
    {4, "P2D2PHASE"}, {5, "P3D2PHASE"}, {6, "P2DBBAR"},  {7, "P3DBBAR"},
    {8, "P2DFS"},     {9, "P3DFS"}};
std::map<std::string, std::string> ParticlePODTypeName = {
    {"P2D", "particles"},
    {"P3D", "particles"},
    {"P2DFLUID", "fluid_particles"},
    {"P3DFLUID", "fluid_particles"},
    {"P2D2PHASE", "twophase_particles"},
    {"P3D2PHASE", "twophase_particles"},
    {"P2DBBAR", "particles"},
    {"P3DBBAR", "particles"},
    {"P2DFS", "particles"},
    {"P3DFS", "particles"}};
}  // namespace mpm

// Particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::Particle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2d("P2D");

// Particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::Particle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3d("P3D");

// Single phase (fluid) particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::FluidParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dfluid("P2DFLUID");

// Single phase (fluid) particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::FluidParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dfluid("P3DFLUID");

// Two-phase particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::TwoPhaseParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2d2phase("P2D2PHASE");

// Two-phase particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::TwoPhaseParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3d2phase("P3D2PHASE");

// Particle2D with B-bar method (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::ParticleBbar<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dbbar("P2DBBAR");

// Particle3D with B-bar method (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::ParticleBbar<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dbbar("P3DBBAR");

// Particle2D with finite strain formulation (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::ParticleFiniteStrain<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dfinitestrain("P2DFS");

// Particle3D with finite strain formulation (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::ParticleFiniteStrain<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dfinitestrain("P3DFS");