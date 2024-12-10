#ifndef MPM_POINT_KELVIN_VOIGT_H_
#define MPM_POINT_KELVIN_VOIGT_H_

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
class PointKelvinVoigt : public PointBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  PointKelvinVoigt(Index id, const VectorDim& coord);

  //! Constructor with id, coordinates and status
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  //! \param[in] status Point status (active / inactive)
  PointKelvinVoigt(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~PointKelvinVoigt() override{};

  //! Delete copy constructor
  PointKelvinVoigt(const PointKelvinVoigt<Tdim>&) = delete;

  //! Delete assignement operator
  PointKelvinVoigt& operator=(const PointKelvinVoigt<Tdim>&) = delete;

  //! Initialise properties
  void initialise() override;

  //! Reinitialise point property
  //! \param[in] dt Time step size
  void initialise_property(double dt) override;

  //! Apply point kelvin voigt constraints
  //! \param[in] dir Direction of kelvin voigt constraint
  //! \param[in] delta Spring vs. Dashpot Weighting Parameter
  //! \param[in] h_min Characteristic length
  //! \param[in] incidence_a Incidence parameter a
  //! \param[in] incidence_b Incidence parameter b
  void apply_point_kelvin_voigt_constraints(
      unsigned dir, double delta, double h_min, double incidence_a,
      double incidence_b) override;

  //! Compute updated position
  //! \param[in] dt Analysis time step
  void compute_updated_position(double dt) noexcept override;

  //! Map point stiffness matrix to cell
  inline bool map_stiffness_matrix_to_cell(double newmark_beta,
                     double newmark_gamma, double dt) override;
  
  //! Map spring stiffness matrix to cell
  inline void map_spring_stiffness_matrix_to_cell();
  
  //! Map dashpot damping matrix to cell
  inline void map_dashpot_damping_matrix_to_cell(double newmark_beta,
                     double newmark_gamma, double dt);

  //! Map enforcement boundary force to node
  //! \param[in] phase Index corresponding to the phase
  void map_boundary_force(unsigned phase) override;

 
  // //! Serialize
  // //! \retval buffer Serialized buffer data
  // std::vector<uint8_t> serialize() override;

  // //! Deserialize
  // //! \param[in] buffer Serialized buffer data
  // void deserialize(const std::vector<uint8_t>& buffer) override;

  //! Assign penalty factor
  //! \param[in] constraint_type Constraint type, e.g. "fixed", "slip"
  //! \param[in] normal_type Normal type, e.g. "cartesian", "assign", "auto"
  //! \param[in] normal_vector Normal vector
  void assign_penalty_parameter(const std::string& constraint_type,
                                double penalty_factor,
                                const std::string& normal_type,
                                const VectorDim& normal_vector) override {
    if (normal_type == "cartesian")
      normal_type_ = mpm::NormalType::Cartesian;
    else if (normal_type == "assign")
      normal_type_ = mpm::NormalType::Assign;
    else if (normal_type == "auto")
      normal_type_ = mpm::NormalType::Automatic;
    normal_ = normal_vector;
  };

  //! Type of point
  std::string type() const override {
    return (Tdim == 2) ? "POINT2DKV" : "POINT3DKV";
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
  //! Delta
  double delta_{std::numeric_limits<double>::epsilon()};
  //! Characteristic length
  double h_min_{1.0};
  //! Incidence a
  double incidence_a_{1.0};
  //! Incidence b
  double incidence_b_{1.0};
  //! Way to obtain normal vector: 0 (Cartesian), 1 (Assign), 2 (Automatic)
  mpm::NormalType normal_type_{mpm::NormalType::Cartesian};
  //! Normal vector
  VectorDim normal_;

};  // PointKelvinVoigt class
}  // namespace mpm

#include "point_kelvin_voigt.tcc"

#endif  // MPM_POINT_KELVIN_VOIGT_H_