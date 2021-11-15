#ifndef MPM_NODAL_PROPERTIES_H_
#define MPM_NODAL_PROPERTIES_H_

#include <Eigen/Dense>
#include <map>

namespace mpm {

// Define Eigen map type
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> MatrixProperty;
typedef Eigen::Map<const MatrixProperty> MapProperty;

// \brief Multimaterial parameters on each node
struct NodalProperties {

  //! Function to create new property with given name and size (rows x cols)
  //! \param[in] property Property name
  //! \param[in] rows Number of nodes times the number of the dimension of the
  //! property (1 if scalar, Tdim if vector)
  //! \param[in] columns Number of materials
  bool create_property(const std::string& property, unsigned rows,
                       unsigned columns);

  //! Initialise all the nodal values for all properties in the property pool
  void initialise_nodal_properties();

  // Return data in the nodal properties map at a specific index
  // \param[in] property Property name
  // \param[in] node_id Id of the node within the property data
  // \param[in] mat_id Id of the material within the property data
  // \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  Eigen::MatrixXd property(const std::string& property, unsigned node_id,
                           unsigned mat_id, unsigned nprops = 1) const;

  // Assign property value to a pair of node and material
  // \param[in] property Property name
  // \param[in] node_id Id of the node within the property data
  // \param[in] mat_id Id of the material within the property data
  // \param[in] property_value Property value to be assigned
  // \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  void assign_property(const std::string& property, unsigned node_id,
                       unsigned mat_id, const Eigen::MatrixXd& property_value,
                       unsigned nprops = 1);

  // Update property value according to a pair of node and material
  // \param[in] property_value Property value to be assigned
  // \param[in] node_id Id of the node within the property data
  // \param[in] mat_id Id of the material within the property data
  // \param[in] nprops Dimension of property (1 if scalar, Tdim if vector)
  // \param[in] property Property name
  void update_property(const std::string& property, unsigned node_id,
                       unsigned mat_id, const Eigen::MatrixXd& property_value,
                       unsigned nprops = 1);

  // Map of properties and their nodal values
  std::map<std::string, Eigen::MatrixXd> properties_;
};  // NodalProperties struct
}  // namespace mpm

#endif  // MPM_NODAL_PROPERTIES_H_
