
## Memory usage

### Blas function

Blas3 level functions use intermediate memory to control CPU cache performance. Blocks of rows/columns
of operand matrices are copied into memory buffer and organized as cache line aligned rows/columns in
ascending memory order. The size of the intermediate buffer is controlled with environment variable
ARMAS_CACHE. The number of rows/columns and their lengths are controlled by BLAS blocking configuration
that is defined in ARMAS_CONFIG variable.

The default intermediate buffer is a thread local global variable. It is automatically created when the
buffer is needed. The main thread registers an exit function that releases the global buffer. The other
threads must handle the release of the global buffer. It is advisable to explicitely request and
release the thread local buffer in the thread main function.

```c
void thread(void *arg) {
    armas_cbuf_create_thread_global();

    /* computations */

    armas_cbuf_release_thread_global();
}
```

Other possiblity is to provide per call pointer to a memory block in conf.work member.

```c
    struct armas_conf cf = *armas_conf_default();
    struct armas_wbuf wbuf;

    /* ff allocation succesfull pointer to wbuf returned otherwise null */
    cf.work = armas_walloc(&wbuf, 128*1024);
```

### LAPACK functions

Some Lapack functions need working space depending on the size of arguments. For these functions
library provides two interfaces: one with suffix '_w' that has explicit workspace argument. Functions
with out the '_w' suffix internal allocate memory per call. The size of the workspace can be
calculated by calling the '_w' interface with workspace size set to zero. The returns with workspace
size containing the required workspace in bytes.



