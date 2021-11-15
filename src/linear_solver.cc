#include "assembler_base.h"
#include "assembler_eigen_implicit.h"
#include "assembler_eigen_semi_implicit_navierstokes.h"
#include "assembler_eigen_semi_implicit_twophase.h"

#include "direct_eigen.h"
#include "iterative_eigen.h"
#include "krylov_petsc.h"
#include "solver_base.h"

// Assembler collections
// Asssembler 2D for Implicit
static Register<mpm::AssemblerBase<2>, mpm::AssemblerEigenImplicit<2>, unsigned>
    assembler_eigen_implicit_2d("EigenImplicit2D");
// Asssembler 3D for Implicit
static Register<mpm::AssemblerBase<3>, mpm::AssemblerEigenImplicit<3>, unsigned>
    assembler_eigen_implicit_3d("EigenImplicit3D");

// Asssembler 2D for NavierStokes
static Register<mpm::AssemblerBase<2>,
                mpm::AssemblerEigenSemiImplicitNavierStokes<2>, unsigned>
    assembler_eigen_semi_implicit_navierstokes_2d(
        "EigenSemiImplicitNavierStokes2D");
// Asssembler 3D for NavierStokes
static Register<mpm::AssemblerBase<3>,
                mpm::AssemblerEigenSemiImplicitNavierStokes<3>, unsigned>
    assembler_eigen_semi_implicit_navierstokes_3d(
        "EigenSemiImplicitNavierStokes3D");

// Asssembler 2D for TwoPhase
static Register<mpm::AssemblerBase<2>,
                mpm::AssemblerEigenSemiImplicitTwoPhase<2>, unsigned>
    assembler_eigen_semi_implicit_twophase_2d("EigenSemiImplicitTwoPhase2D");
// Asssembler 3D for TwoPhase
static Register<mpm::AssemblerBase<3>,
                mpm::AssemblerEigenSemiImplicitTwoPhase<3>, unsigned>
    assembler_eigen_semi_implicit_twophase_3d("EigenSemiImplicitTwoPhase3D");

// Linear Solver collections
// Eigen SparseLU
static Register<mpm::SolverBase<Eigen::SparseMatrix<double>>,
                mpm::DirectEigen<Eigen::SparseMatrix<double>>, unsigned, double>
    solver_direct_eigen("DirectEigen");

// Eigen Conjugate Gradient
static Register<mpm::SolverBase<Eigen::SparseMatrix<double>>,
                mpm::IterativeEigen<Eigen::SparseMatrix<double>>, unsigned,
                double>
    solver_iterative_eigen("IterativeEigen");

// Krylov Methods PTSC
#ifdef USE_PETSC
static Register<mpm::SolverBase<Eigen::SparseMatrix<double>>,
                mpm::KrylovPETSC<Eigen::SparseMatrix<double>>, unsigned, double>
    solver_krylov_petsc("KrylovPETSC");
#endif