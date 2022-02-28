#ifndef MPM_DISCONTINUITY_3D_H_
#define MPM_DISCONTINUITY_3D_H_

#include "discontinuity_base.h"

//! MPM namespace
namespace mpm {
//! Discontinuity3D class derived from DiscontinuityBase class for 3D
template <unsigned Tdim>
class Discontinuity3D : public DiscontinuityBase<Tdim> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor
  //! \param[in] discontinuity_props discontinuity properties
  //! \param[in] discontinuity id
  Discontinuity3D(const Json& discontinuity_props, unsigned id);

  //! Constructor
  //! \param[in] discontinuity id
  //! \param[in] initiation properties
  Discontinuity3D(unsigned id,
                  std::tuple<double, double, double, double, double, int, bool,
                             bool>& initiation_property);

  //! Initialization
  //! \param[in] the coordinates of all points
  //! \param[in] the point index of each surface
  virtual bool initialise(const std::vector<VectorDim>& points,
                          const std::vector<std::vector<mpm::Index>>& surfs);
  //! Create elements from file
  //! \param[in] surfs the point index list of each surface
  virtual bool create_surfaces(
      const std::vector<std::vector<mpm::Index>>& surfs) override;

  //! Initialize the center and normal vector of each surface
  bool initialise_center_normal();

  //! Return the cross product of ab and bc
  //! \param[in] a,b,c coordinates of three points
  //! \retval the cross product of ab and bc
  VectorDim three_cross_product(const VectorDim& a, const VectorDim& b,
                                const VectorDim& c);

  // Return the levelset values of each coordinates
  //! \param[in] coordinates coordinates
  //! \param[in] phi_list the reference of phi for all coordinates
  void compute_levelset(const VectorDim& coordinates,
                        double& phi_particle) override;

  //! Compute the normal vectors of coordinates
  //! \param[in] coordinates The coordinates
  //! \param[in] normal vector the normal vector of the given coordinates
  void compute_normal(const VectorDim& coordinates,
                      VectorDim& normal_vector) override;

  //! Assign point friction coefficient
  void assign_point_friction_coef() noexcept override;

  //! Compute updated position
  //! \param[in] dt Time-step
  void compute_updated_position(const double dt) noexcept;

 protected:
  //! Id
  using mpm::DiscontinuityBase<Tdim>::id_;
  //! Vector of points
  using mpm::DiscontinuityBase<Tdim>::points_;
  //! Logger
  using mpm::DiscontinuityBase<Tdim>::console_;
  //! Friction coefficient
  using mpm::DiscontinuityBase<Tdim>::friction_coef_;
  //! Width
  using mpm::DiscontinuityBase<Tdim>::width_;
  //! The mark points move with which side
  using mpm::DiscontinuityBase<Tdim>::move_direction_;

 private:
  //! Vector of surfaces
  std::vector<discontinuity_surface<Tdim>> surfaces_;
};

}  // namespace mpm
#include "discontinuity_3d.tcc"

#endif  // MPM_HEXAHEDRON_ELEMENT_H_
