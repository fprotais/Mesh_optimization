# Versatile Volume Fitting with Automatic Feature Preservation

This is the reference implementation of the optimization approach presented in

Versatile Volume Fitting with Automatic Feature Preservation (link to come) <br />
[F. Protais](https://fprotais.github.io/), [G. Cherchi](https://www.gianmarcocherchi.com), [M. Livesu](http://pers.ge.imati.cnr.it/livesu/)

> ⚠️ This code is intended to be included in [CGAL](https://github.com/cgal/cgal) in the near future. As such, it may undergo noticeable changes in its structure. This repository will always remain as a link to the CGAL version, and hopefully provide python bindings.

The core of the code (`Mesh_optimization` directory) is header only and templated, hence fully agnostic to mesh representation. The target is to deploy a toolbox that can be used with any application with limited linking, relying solely on [Eigen3](https://libeigen.gitlab.io) as an external tool. We accessed the functionalities of the solver through the following popular geometry processing libraries, which **entirely** take care of input/output, mesh storage and handling:
* [Geogram](https://github.com/BrunoLevy/geogram)
* [Cinolib](https://github.com/mlivesu/cinolib)
* [LibIGL](https://github.com/libigl/libigl)
* [CGAL](https://github.com/CGAL/cgal)

We have plans to extend this list to [OpenVolumeMesh](https://github.com/OpenVolumeMesh/OpenVolumeMesh), [GMDS](https://github.com/LIHPC-Computational-Geometry/gmds) and [ultimaille](https://github.com/ultimaille/ultimaille). You want your data structure on this list? Let us know.

We provide various examples from the paper:
* [**Mesh projection and smoothing**](#mesh-smoothing-and-projection) with Geogram or Cinolib
* [**Handle based deformation**](#handle-based-deformation) with LibIGL
* [**Offset computation with Alpha-wrapper**](#mesh-offsetting) with CGAL

We will slowly expand this list to match all figures in the paper.  <!-- for the [Graphics Replicability Stamp Initiative](https://www.replicabilitystamp.org/#). -->

For building instructions, see [Installation section](#installation). For visualization of meshes, see section [Volume visualization](#volume-visualization).

## Element orientations

Element orientation is key to optimize mesh quality (and guarantee validity). Not all available models and tools stick to the same convention, sometimes generating unexplainable failures in the software. We stick to the VTK convention for all mixed elements (see Figure 2 [here](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html)).
Notice that due to our untangling capabilities, we cannot autonomously guess or fix input element orientation. Cinolib and Geogram examples contain a `void set_orientation(bool inv_tet, bool inv_hex, bool inv_pyr, bool inv_wed)` function to help on that aspect but, as a rule of thumb, input meshes that stick to a different convention must be fixed prior to calling our solver.

# Installation

## Compiling

Compilation on Unix-like devices can be done as follows:
```Bash
mkdir build;
cd build;
cmake ..;
make -j;
```
CMake will automatically fetch the corresponding libs and link it to the executables. Executables for all examples will be created and installed in a dedicated `/build/bin` directory.

The code is tested to work on recent Windows, Linux and MacOS. But we leave non-linux installation to users. **Note that OpenMP will provide significant performance gain, but it is non trivial to include on MacOS**.

## Requirements

The only general requirement is [Eigen3](https://libeigen.gitlab.io/), which will be accessed through cmake:
```Bash
find_package(Eigen3 3.3.0 QUIET REQUIRED)
```
Similarly, the [CGAL](https://github.com/CGAL/cgal)  library must be installed independently and will be accessed by:
```Bash
find_package(CGAL REQUIRED)
```
To avoid this requirement, it is possible to build the project with CGAL disabled:
```Bash
cmake .. -DBUILD_CGAL=FALSE
```

# Mesh smoothing and projection

This executable is the most standard use of the code: improve the quality of mesh and fit to its geometric target. The code can be called as:
```Bash
./bin/geogram_smooth ../data/sphere.mesh ../data/max-planck.obj
```
*sphere.mesh* represent the volumetric tetrahedral mesh and *max-planck.obj* the geometric target represented as a surface triangle mesh. The code works with mesh containing Tetrahedra, Pyramids, Wedges and Hexahedra.

The same executable can also be called with a unique input parameter, as:
```Bash
./bin/geogram_smooth ../data/fandisk_kenshi_hexmesh.mesh
```
In this case it will operate as a smoothing (or untangling algorithm), improving the mesh quality while trying to preserve the outer boundary.

In both cases, the result will be saved as `output.mesh`. The executable `cinolib_smooth` operates in the exact same way and only differentiates from `geogram_smooth` because it uses a different frontend for mesh handling.

# Handle based deformation

This executable will perform a deformation of the input object to match a set of input constraints, as shown in Fig. 12.
```Bash
./bin/libigl_handle_deformation ../data/bunny.mesh ../data/bunny_handles.txt
```
*bunny.mesh* is a standard tetrahedral mesh (as supported by LibIGL) and *bunny_handles.txt* contains vertex handle constraints as: `id x y z`.

Result will be saved as `handle_output.mesh`.

# Mesh offsetting

This executable can be used to reproduce Feature preserving offsets presented in the paper (Fig 14. and Fig 15.) We rely on the [Alpha-wrapper](https://doc.cgal.org/latest/Alpha_wrap_3/index.html) proposed by the CGAL library and use their parameters as input:
```Bash
./bin/cgal_offset_computation ../data/stairs.off 20 600
```
The input mesh must in the [OFF format](https://en.wikipedia.org/wiki/OFF_(file_format)) due to CGAL requirements. X=20 is used to compute the gate size in Alpha-wrap as alpha=D/X with D the diagonal of the bbox of the object. Y=600 is used to compute the offset of the mesh with delta=D/Y.

The executable will output two meshes: `offset_wrapper.mesh` and `offset_ours.mesh` to compare the before/after.

The parameters to run the different examples are as follows:
* Fig 14:
```Bash
./bin/cgal_offset_computation ../data/stairs.off 20 600
./bin/cgal_offset_computation ../data/stairs.off 40 600
./bin/cgal_offset_computation ../data/stairs.off 60 600
./bin/cgal_offset_computation ../data/stairs.off 80 600
```
* Fig 15:
```Bash
./bin/cgal_offset_computation ../data/43149.off 150 100 #  1% offset - slow remeshing (larger offset means slower)
./bin/cgal_offset_computation ../data/43149.off 150 50  #  2% offset
./bin/cgal_offset_computation ../data/43149.off 150 20  #  5% offset
./bin/cgal_offset_computation ../data/43149.off 150 10  # 10% offset
```
The main bottleneck of the code is the inside remeshing that we perform with CGAL after the Alpha-wrapper.

# Volume visualization

Meshes can be visualized using various tools depending on their format. [GraphiteThree](https://github.com/BrunoLevy/GraphiteThree) is capable of displaying all meshes provided in Data and generated by our executables, and any file handled by `geogram_smooth`.

# License

This code is under AGPL and shall not be distributed in or with closed source software. For commercial uses, alternative licensing will be provided after inclusion into the CGAL library.

# Citing this repo

Soon to come. 

