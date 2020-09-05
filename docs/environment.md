
## Environment variables

Following environment variables can be used to configure aspects of execution

* ARMAS_CONFIG = mb,kb,nb,lb,b1min,b2min,fixed

    Defines the blocking configuration of the library functions. First three values
    (*mb*, *kb*, *nb*) define the matrix kernel intermediate block size. These values are
    relative values and final sizes are computed based on *memsize* value defined
    in *ARMAS_CACHE* variable or the memory block size of per call intermediate memory.

    The *lb* value is Lapack implementations blocking size for blocked algorithms.

    The values of *b1min* and *b2min* the blocking size for BLAS level1 and BLAS level2
    recursive implementations.

    The *fixed* is boolean indicating that (mb,kb,nb) triplet values are absolute
    values. It is responsibility of the caller to ensure that provided intermediate
    memory if large enough.

    All the numeric values indicate are the number of the elements.

* ARMAS_CACHE = memsize,l1size

    These values define the intermediate memory space of matrix kernel. Sizes are in bytes.
    The *memsize* defines the total allocated memory that will hold parts of operand
    matrices in cache. The *l1size* affectes internal looping in the matrix kernel. It
    strictly smaller the *memsize*.

* ARMAS_AC_CONFIG = min_elems,maxproc,policy,cpuspec

    This variable defines configuration the library default BLAS accelerators. The *min_elems*
    defines the threshold for using the accelerator. The *maxproc* is max number of threads to
    use. The *policy* is accelelator implementation spefic scheduling policy configuration. And
    the *cpuspec* is CPU pinning configuration for some accelerators.



