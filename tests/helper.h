
#ifndef __TEST_HELPER_H
#define __TEST_HELPER_H

extern double zero(int i, int j);
extern double one(int i, int j);
extern double zeromean(int i, int j);
extern double unitrand(int i, int j);
extern double rowno(int i, int j);
extern double colno(int i, int j);

extern double time_msec();
extern double gflops(double ms, int64_t count);  
extern void flush();

extern int check(armas_d_dense_t *A, armas_d_dense_t *B, int chkdir, const char *msg);


#endif
