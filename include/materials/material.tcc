//! Get material property
template <unsigned Tdim>
template <typename Ttype>
Ttype mpm::Material<Tdim>::property(const std::string& key) {
  try {
    return properties_[key].template get<Ttype>();
  } catch (std::exception& except) {
    console_->error("Property call to material parameter not found: {}",
                    except.what());
    throw std::runtime_error(
        "Property call to material parameter not found or invalid type");
  }
}

template <unsigned Tdim>
template <typename Ttype>
bool mpm::Material<Tdim>::contain_property(const std::string& key) {
  bool exists = false;
  try {
    properties_[key].template get<Ttype>();
    exists=true;
  } catch (std::exception& except) {
    exists=false;
  }
  return exists;
}