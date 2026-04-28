#ifndef MPM_POINT_JOYNER_CHEN_H_
#define MPM_POINT_JOYNER_CHEN_H_

// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "point_base.h"

namespace mpm {

// Forward declaration of Material
template <unsigned Tdim>
class Material;

//! Point class to impose nonconforming Kelvin Voigt BC
//! \tparam Tdim Dimension
template <unsigned Tdim>
class PointJoynerChen : public PointBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  PointJoynerChen(Index id, const VectorDim& coord);

  //! Constructor with id, coordinates and status
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  //! \param[in] status Point status (active / inactive)
  PointJoynerChen(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~PointJoynerChen() override {};

  //! Delete copy constructor
  PointJoynerChen(const PointJoynerChen<Tdim>&) = delete;

  //! Delete assignement opera+tor
  PointJoynerChen& operator=(const PointJoynerChen<Tdim>&) = delete;

  //! Initialise properties
  void initialise() override;

  //! Apply point joyner chen constraints
  //! \param[in] velocity Velocity ground motion constraint
  void assign_joyner_chen_constraints(unsigned dir, double velocity) override;

  //! Compute updated position
  //! \param[in] dt Analysis time step
  void compute_updated_position(
      double dt, unsigned phase, double blending_ratio = 1.0,
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::APIC) noexcept override;

  //! Map dashpot damping matrix to cell
  inline bool map_damping_matrix_to_cell(double newmark_gamma, double newmark_beta,
                                        double dt) override;

  //! Map enforcement boundary force to node
  //! \param[in] phase Index corresponding to the phase
  void map_boundary_force(unsigned phase) override;

  // //! Serialize
  // //! \retval buffer Serialized buffer data
  // std::vector<uint8_t> serialize() override;

  // //! Deserialize
  // //! \param[in] buffer Serialized buffer data
  // void deserialize(const std::vector<uint8_t>& buffer) override;

  //! Assign point properties
  //! \param[in] scalar_properties Map of scalar properties
  //! \param[in] vector_properties Map of vector properties
  void assign_properties(const std::map<std::string, double>& scalar_properties,
                         const std::map<std::string, std::vector<double>>&
                             vector_properties) override;

  //! Reinitialise point property
  //! \param[in] dt Time step size
  void initialise_properties(double dt) override;

  //! Type of point
  std::string type() const override {
    return (Tdim == 2) ? "POINT2DJC" : "POINT3DJC";
  }

  //  protected:
  //   //! Compute pack size
  //   //! \retval pack size of serialized object
  //   int compute_pack_size() const override;

 protected:
  //! point id
  using PointBase<Tdim>::id_;
  //! coordinates
  using PointBase<Tdim>::coordinates_;
  //! Status
  using PointBase<Tdim>::status_;
  //! Cell
  using PointBase<Tdim>::cell_;
  //! Cell id
  using PointBase<Tdim>::cell_id_;
  //! Nodes
  using PointBase<Tdim>::nodes_;
  //! Shape functions
  using PointBase<Tdim>::shapefn_;
  //! Displacement
  using PointBase<Tdim>::displacement_;
  //! Area
  using PointBase<Tdim>::area_;
  //! Pack size
  using PointBase<Tdim>::pack_size_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Young's modulus
  double youngs_modulus_{0.0};
  //! Density
  double density_{0.0};
  //! Poisson's ratio
  double poisson_ratio_{0.0};
  //! Normal vector
  VectorDim normal_;
  //! Imposed velocity
  VectorDim imposed_velocity_;
  //! Constraint flags: 1 = constrained, 0 = unconstrained, per direction
  Eigen::Matrix<int, Tdim, 1> constraint_flags_;

};  // PointJoynerChen class
}  // namespace mpm

#include "point_joyner_chen.tcc"

#endif  // MPM_POINT_JOYNER_CHEN_H_