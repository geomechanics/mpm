#include "vtk_writer.h"

//! VTK Writer class Constructor with coordniates
//! \param[in] coordinate Point coordinates
//! \param[in] node_pairs Node ID pairs to form elements
VtkWriter::VtkWriter(
    const std::vector<Eigen::Matrix<double, 3, 1>>& coordinates) {
  // Assign points
  points_ = vtkSmartPointer<vtkPoints>::New();
  unsigned long long id = 0;
  for (const auto& coordinate : coordinates) {
    const double* point = coordinate.data();
    points_->InsertPoint(id, point);
    ++id;
  }
}

//! Write coordinates
//! \param[in] filename Output file to write geometry
void VtkWriter::write_geometry(const std::string& filename) {

  // Create a polydata to store everything in it
  auto pdata = vtkSmartPointer<vtkPolyData>::New();

  // Add the points to the dataset
  pdata->SetPoints(points_);

  // Write file
  auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

  writer->SetFileName(filename.c_str());

  writer->SetDataModeToBinary();

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(pdata);
#else
  writer->SetInputData(pdata);
#endif

  writer->Write();
}

//! \brief Write vector data
//! \param[in] filename Output file to write geometry
//! \param[in] data Vector data
//! \param[in] data_field Field name ("Displacement", "Forces")
void VtkWriter::write_vector_point_data(
    const std::string& filename, const std::vector<Eigen::Vector3d>& data,
    const std::string& data_field) {

  // Create a polydata to store everything in it
  auto pdata = vtkSmartPointer<vtkPolyData>::New();

  // Add the points to the dataset
  pdata->SetPoints(points_);

  // Create an array to hold distance information
  auto vectordata = vtkSmartPointer<vtkDoubleArray>::New();
  vectordata->SetNumberOfComponents(3);
  vectordata->SetNumberOfTuples(pdata->GetNumberOfPoints());
  vectordata->SetName(data_field.c_str());

  // Evaluate the signed distance function at all of the grid points
  for (vtkIdType id = 0; id < pdata->GetNumberOfPoints(); ++id) {
    const double* vdata = data.at(id).data();
    vectordata->SetTuple(id, vdata);
  }

  // Add the SignedDistances to the grid
  pdata->GetPointData()->SetVectors(vectordata);

  // Write file
  auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

  writer->SetFileName(filename.c_str());

  writer->SetDataModeToBinary();

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(pdata);
#else
  writer->SetInputData(pdata);
#endif

  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();
}

//! VTK Write mesh
void VtkWriter::write_mesh(
    const std::string& filename,
    const std::vector<Eigen::Vector3d>& coordinates,
    const std::vector<std::array<mpm::Index, 2>>& node_pairs) {
  // Assing points
  auto points = vtkSmartPointer<vtkPoints>::New();
  unsigned id = 0;
  for (const auto& coordinate : coordinates) {
    const double* point = coordinate.data();
    points->InsertPoint(id, point);
    ++id;
  }
  // Assign elements
  // Create a cell array to store the lines
  auto lines = vtkSmartPointer<vtkCellArray>::New();

  for (const auto& node_pair : node_pairs) {
    // Create a line between the two points
    auto line = vtkSmartPointer<vtkLine>::New();
    // the SetId(A,B) call is the following
    line->GetPointIds()->SetId(0, node_pair[0]);
    // A = the id of the point relative to the line - this can only be 0 or 1.
    // B = the index into the vtkPoints object of the point that you would
    // like to set the Ath point to.
    line->GetPointIds()->SetId(1, node_pair[1]);

    lines->InsertNextCell(line);
  }
  // Create a polydata to store everything in it
  auto pdata = vtkSmartPointer<vtkPolyData>::New();

  // Add the points to the dataset
  pdata->SetPoints(points);

  // Add the lines to the dataset
  pdata->SetLines(lines);

  // Write file
  auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

  writer->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(pdata);
#else
  writer->SetInputData(pdata);
#endif
  writer->Write();
}

//! Write Parallel VTK file
void VtkWriter::write_parallel_vtk(const std::string& filename,
                                   const std::string& attribute, int mpi_size,
                                   unsigned step, unsigned max_steps) {

  std::string ppolydata =
      "<?xml version=\"1.0\"?>\n<VTKFile type=\"PPolyData\" version=\"0.1\" "
      "byte_order=\"LittleEndian\" "
      "compressor=\"vtkZLibDataCompressor\">\n<PPolyData "
      "GhostLevel=\"0\">\n\t<PPointData Vectors=\"" +
      attribute +
      "\">\n\t\t<PDataArray "
      "type=\"Float64\" Name=\"" +
      attribute +
      "\" "
      "NumberOfComponents=\"3\"/>\n\t</"
      "PPointData>\n\n\t<PPoints>\n\t\t<PDataArray "
      "type=\"Float32\" Name=\"Points\" "
      "NumberOfComponents=\"3\"/>\n\t</PPoints>\n";

  for (unsigned i = 0; i < mpi_size; ++i) {
    std::stringstream file_name;
    file_name.str(std::string());
    file_name << attribute;
    const std::string rank_size =
        "-" + std::to_string(i) + "_" + std::to_string(mpi_size) + "-";
    file_name << rank_size;

    file_name.fill('0');
    int digits = log10(max_steps) + 1;
    file_name.width(digits);
    file_name << step;
    file_name << ".vtp";
    ppolydata += "\n\t<Piece Source=\"" + file_name.str() + "\"/>";
  }
  ppolydata += "\n</PPolyData>\n\n</VTKFile>";

  // Write parallel VTK file
  ofstream pvtk;
  pvtk.open(filename);
  pvtk << ppolydata;
  pvtk.close();
}