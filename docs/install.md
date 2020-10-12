
## Installing

### Building from GIT sources.

Need autoconf, automake and libtool packages.

```sh
   $ ./bootstrap.sh
   $ mkdir build
   $ cd build; ../configure CFLAGS=-O3
   $ make
   $ make check
   $ make install
```

Building documentation. Needs doxygen package.

```sh
   $ cd build/docs
   $ make docs
   $ doxygen
```

### Configuration options

   * \--enable-float64

     Compile double precision version, enabled by default. Tested.

   * \--enable-accelerators

     Enable accelerator API. Includes threaded accelerators for BLAS3 functions.

   * \--enable-ext-precision

     Extended precison BLAS functionality. Enabled by default.

   * \--enable-sparse

     Enable sparse matrix support. Includes iterative solvers. No support for direct
     methods.

   * \--enable-compat

     Add compability layer; fortran and cblas functions, disabled by default
	(Proof of concept and not tested)

   * \--enable-float32

     Compile single precision version, disabled by default.


### SIMD optimizations

   SIMD accelerated versions of internal kernel functions exist for Intel SSE, AVX and FMA
   instruction sets. These exist only for double precision real versions. Currently these are
   automatically enabled when library is compiled (GCC option -march=native)
