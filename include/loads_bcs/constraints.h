#ifndef MPM_CONSTRAINTS_H_
#define MPM_CONSTRAINTS_H_

#include <memory>

#include "absorbing_constraint.h"
#include "acceleration_constraint.h"
#include "adhesion_constraint.h"
#include "displacement_constraint.h"
#include "friction_constraint.h"
#include "logger.h"
#include "mesh.h"
#include "pressure_constraint.h"
#include "velocity_constraint.h"

namespace mpm {

//! Constraints class to store velocity and frictional constraints
//! \brief Constraint class to store a constraint on mesh
template <unsigned Tdim>
class Constraints {
 public:
  // Constructor with mesh as input argument
  Constraints(std::shared_ptr<mpm::Mesh<Tdim>> mesh) {
    mesh_ = mesh;
    console_ =
        std::make_unique<spdlog::logger>("Constraints", mpm::stdout_sink);
  }

  //! Assign nodal acceleration constraints
  //! \param[in] setid Node set id
  //! \param[in] acceleration_constraints Accelerartion constraint at node, dir,
  //! acceleration
  bool assign_nodal_acceleration_constraint(
      int set_id,
      const std::shared_ptr<mpm::AccelerationConstraint>& constraint);

  //! Assign acceleartion constraints to nodes
  //! \param[in] acceleration_constraints Constraint at node, dir, and
  //! acceleration
  bool assign_nodal_acceleration_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, double>>&
          acceleration_constraints);

  //! Assign nodal velocity constraints
  //! \param[in] setid Node set id
  //! \param[in] velocity_constraints Velocity constraint at node, dir, velocity
  bool assign_nodal_velocity_constraint(
      int set_id, const std::shared_ptr<mpm::VelocityConstraint>& constraint);

  //! Assign velocity constraints to nodes
  //! \param[in] velocity_constraints Constraint at node, dir, and velocity
  bool assign_nodal_velocity_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, double>>&
          velocity_constraints);

  //! Assign nodal frictional constraints
  //! \param[in] setid Node set id
  //! \param[in] friction_constraints Constraint at node, dir, sign_n, friction
  bool assign_nodal_frictional_constraint(
      int nset_id,
      const std::shared_ptr<mpm::FrictionConstraint>& fconstraints);

  //! Assign friction constraints to nodes
  //! \param[in] friction_constraints Constraint at node, dir, sign_n, friction
  bool assign_nodal_friction_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, int, double>>&
          friction_constraints);

  //! Assign nodal adhesional constraints
  //! \param[in] setid Node set id
  //! \param[in] adhesion_constraints Constraint at node, dir, sign_n, adhesion,
  //! h_min, nposition
  bool assign_nodal_adhesional_constraint(
      int nset_id,
      const std::shared_ptr<mpm::AdhesionConstraint>& aconstraints);

  //! Assign adhesion constraints to nodes
  //! \param[in] adhesion_constraints Constraint at node, dir, sign_n, adhesion,
  //! h_min, nposition
  bool assign_nodal_adhesion_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, int, double, double,
                                   int>>& adhesion_constraints);

  //! Assign nodal pressure constraints
  //! \param[in] mfunction Math function if defined
  //! \param[in] setid Node set id
  //! \param[in] phase Index corresponding to the phase
  //! \param[in] pconstraint Pressure constraint at node
  bool assign_nodal_pressure_constraint(
      const std::shared_ptr<FunctionBase>& mfunction, int set_id,
      unsigned phase, double pconstraint);

  //! Assign nodal pressure constraints to nodes
  //! \param[in] pressure_constraints Constraint at node, pressure
  bool assign_nodal_pressure_constraints(
      const unsigned phase,
      const std::vector<std::tuple<mpm::Index, double>>& pressure_constraints);

  //! Assign nodal absorbing constraints
  //! \param[in] setid Node set id
  //! \param[in] absorbing_constraints Constraint at node, dir, delta, h_min, a,
  //! b, and position
  bool assign_nodal_absorbing_constraint(
      int nset_id,
      const std::shared_ptr<mpm::AbsorbingConstraint>& absorbing_constraints);

  //! Assign absorbing constraints to nodes
  //! \param[in] absorbing_constraints Constraint at node, dir, delta, h_min, a,
  //! b, and position
  bool assign_nodal_absorbing_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, double, double, double,
                                   double, mpm::Position>>&
          absorbing_constraints);

  //! Assign absorbing constraints pointers and ids
  //! \param[in] nset_id Node set IDs
  //! \param[in] absorbing_constraint Constraint at node
  void assign_absorbing_id_ptr(
      unsigned nset_id,
      std::shared_ptr<mpm::AbsorbingConstraint>& absorbing_constraint);

  //! Absorbing constraint
  std::vector<std::shared_ptr<mpm::AbsorbingConstraint>> absorbing_ptrs()
      const {
    return absorbing_constraint_;
  }

  //! Absorbing constraint node set IDs
  std::vector<unsigned> absorbing_ids() const { return absorbing_nset_id_; }

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Assign nodal displacement constraints for implicit solver
  //! \ingroup Implicit
  //! \param[in] setid Node set id
  //! \param[in] displacement_constraints Displacement constraint at node, dir,
  //! velocity
  bool assign_nodal_displacement_constraint(
      const std::shared_ptr<FunctionBase>& dfunction, int set_id,
      const std::shared_ptr<mpm::DisplacementConstraint>& dconstraint);

  //! Assign displacement constraints to nodes
  //! \ingroup Implicit
  //! \param[in] displacement_constraints Constraint at node, dir, and velocity
  bool assign_nodal_displacement_constraints(
      const std::vector<std::tuple<mpm::Index, unsigned, double>>&
          displacement_constraints);
  /**@}*/

 private:
  //! Mesh object
  std::shared_ptr<mpm::Mesh<Tdim>> mesh_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! List of absorbing constraint ptrs
  std::vector<std::shared_ptr<mpm::AbsorbingConstraint>> absorbing_constraint_;
  //! List of absorbing constraint node set Ids
  std::vector<unsigned> absorbing_nset_id_;
};
}  // namespace mpm

#include "constraints.tcc"

#endif  // MPM_CONSTRAINTS_H_
