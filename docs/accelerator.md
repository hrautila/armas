
## Accelerators

Accelerators define a simple API to implement threaded versions of library functions. Three accelerator
implementations are included.

An accelerator is created with an explicit call to *armas_ac_init()* where first parameter is pointer
to an *armas_ac_handle_t* and second parameter the scheduler name. Initialized handle is saved into
the *armas_conf_t* member *accel*.

Work is dispatched to accelerator if the BLAS function called supports accelerators and accelerator
is provided in configuration block.

User is responsible for closing the accelerator by call to *armas_ac_release()*.

### Simple

Simple accelerator ARMAS_AC_SIMPLE uses at most *maxproc* threads for computation. Recursive calls
are used for thread housekeeping.

### Transient

Accelerator ARMAS_AC_TRANSIENT uses transient threads with *blocked* or *tiled* scheduling. In **blocked**
scheduling the work is divided into at most *maxproc* blocks and scheduled to at most *maxproc* worker
threads. The transient *blocked* scheduling is like *Simple* scheduling except for different thread
housekeeping methods. 

The *tiled* scheduler divides work to fixed size tiles and schedules them to at most *maxproc* threads
for execution. The tile size is defined with *min_elems* in ARMAS_AC_CONFIG variable.

### Workers

The ARAMS_AC_WORKER accelerator uses persistent worker threads for execution. The worker threads are
created on first work dispatch and persist until accelerator is released. Worker threads may be pinned
on selected CPU cores. Accelerator support *blocked* and *tiled* scheduling.
