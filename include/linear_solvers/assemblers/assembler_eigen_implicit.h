#ifndef MPM_ASSEMBLER_EIGEN_IMPLICIT_H_
#define MPM_ASSEMBLER_EIGEN_IMPLICIT_H_

#include <Eigen/Sparse>
#include <string>

// Speed log
#include "assembler_base.h"
#include "spdlog/spdlog.h"

#include "mesh.h"

namespace mpm {
template <unsigned Tdim>
class AssemblerEigenImplicit : public AssemblerBase<Tdim> {
 public:
  //! Constructor
  //! \param[in] node_neighbourhood Number of node neighbourhood considered
  AssemblerEigenImplicit(unsigned node_neighbourhood);

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Return stiffness matrix
  //! \ingroup Implicit
  Eigen::SparseMatrix<double>& stiffness_matrix() override {
    return stiffness_matrix_;
  }

  //! Assemble stiffness matrix
  //! \ingroup Implicit
  bool assemble_stiffness_matrix() override;

  //! Return residual force RHS vector
  //! \ingroup Implicit
  Eigen::VectorXd& residual_force_rhs_vector() override {
    return residual_force_rhs_vector_;
  }

  //! Assemble residual force RHS vector
  //! \ingroup Implicit
  bool assemble_residual_force_right() override;

  //! Assign displacement constraints
  //! \ingroup Implicit
  bool assign_displacement_constraints(double current_time) override;

  //! Apply displacement constraints to equilibrium equation
  //! \ingroup Implicit
  void apply_displacement_constraints() override;

  //! Return displacement increment
  //! \ingroup Implicit
  Eigen::VectorXd& displacement_increment() override {
    return displacement_increment_;
  }

  //! Assign displacement increment
  //! \ingroup Implicit
  void assign_displacement_increment(
      const Eigen::VectorXd& displacement_increment) override {
    displacement_increment_ = displacement_increment;
  }

  /**@{*/

  /**
   * \defgroup Thermal Functions forthermo-mechancial coupling MPM
   */
  /**@{*/
  //! Assemble thermal expansivity matrix
  bool assemble_thermal_expansivity_matrix() override;

  //! Assemble thermal conductivity matrix
  bool assemble_thermal_conductivity_matrix() override;

  // Assemble residual heat right vector
  bool assemble_residual_heat_right() override;

  //! Assemble global stiffness matrix
  bool assemble_global_stiffness_matrix() override;

  //! return global stiffness matrix
  Eigen::SparseMatrix<double>& global_stiffness_matrix() {
    return global_stiffness_matrix_;
  }

  //! Assemble global residual right
  bool assemble_global_residual_right() override;

  //! Return global residual rhs vector
  Eigen::VectorXd& global_residual_rhs_vector() {
    return global_residual_rhs_vector_;
  }

  //! Assign temperature constraints
  bool assign_temperature_constraints(double current_time) override;

  //! Apply temperature constraints vector
  void apply_temperature_increment_constraints() override;

  //! Apply temperature constraints vector
  void apply_coupling_constraints() override; 

  //! Assign displacement and temperature increment
  void assign_solution_increment(
      const Eigen::VectorXd& solution_increment) override {
    solution_increment_ = solution_increment;
    // Extract displacement increment (first N * Tdim elements)
    displacement_increment_ = solution_increment.head(active_dof_ * Tdim);
    // Extract temperature increment (last N elements)
    temperature_increment_ = solution_increment.tail(active_dof_); 
  }

  //! Return solution increment
  Eigen::VectorXd& solution_increment() override {
    return solution_increment_;
  }

  //! Assign temperature increment
  void assign_temperature_increment(
      const Eigen::VectorXd& temperature_increment) override {
    temperature_increment_ = temperature_increment;
  }

  //! Return temperature increment
  Eigen::VectorXd& temperature_increment() override {
    return temperature_increment_;
  }
  /**@{*/
  

 protected:
  //! number of nodes
  using AssemblerBase<Tdim>::active_dof_;
  //! Mesh object
  using AssemblerBase<Tdim>::mesh_;
  //! Number of sparse matrix container size
  using AssemblerBase<Tdim>::sparse_row_size_;
  //! Global node indices
  using AssemblerBase<Tdim>::global_node_indices_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Stiffness matrix
  Eigen::SparseMatrix<double> stiffness_matrix_;
  //! Residual force RHS vector
  Eigen::VectorXd residual_force_rhs_vector_;
  //! Displacement constraints
  Eigen::SparseVector<double> displacement_constraints_;
  //! Displacement increment
  Eigen::VectorXd displacement_increment_;

  //! Thermal stiffness matrix
  Eigen::SparseMatrix<double> thermal_expansivity_matrix_;
  //! Heat laplacian matrix
  Eigen::SparseMatrix<double> thermal_conductivity_matrix_;
  //! Global matrix for TM coupling equation
  Eigen::SparseMatrix<double> global_stiffness_matrix_;
  //! Residual heat RHS vector
  Eigen::VectorXd residual_heat_rhs_vector_;
  //! Residual TM coupling RHS vector
  Eigen::VectorXd global_residual_rhs_vector_;  
  //! Temperatue constraints
  Eigen::SparseVector<double> temperature_increment_constraints_;
  //! Temperature increment
  Eigen::VectorXd temperature_increment_;
  //! Temperature increment
  Eigen::VectorXd solution_increment_;  
  /**@{*/
};  // namespace mpm
}  // namespace mpm

#include "assembler_eigen_implicit.tcc"
#include "assembler_eigen_implicit_thermal.tcc"
#endif  // MPM_ASSEMBLER_EIGEN_IMPLICIT_H_
