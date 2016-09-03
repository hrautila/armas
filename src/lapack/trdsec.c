
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

//! \file
//! Secular function

#include "dtype.h"
#include "dlpack.h"

// ------------------------------------------------------------------------------
// this file provides following type independet functions
#if defined(armas_x_trdsec_solve) && defined(armas_x_trdsec_eigen) && \
    defined(armas_x_trdsec_solve_vec)
#define __ARMAS_PROVIDES 1
#endif
// this file requires external public functions
#if defined(armas_x_discriminant) 
#define __ARMAS_REQUIRES 1
#endif

// References
// (1) Ren-Chang Li
//         Solving Secular Equations Stably and Efficiently, 1993 (LAWN #93)
// (2) Demmel
//         Applied Numerical Linear Algebra, 1996, (section 5.3.3)
// (3) Gu, Eisenstat
//         A Stable and Efficient Algorithm for the Rank-one Modification of the
//         Symmetric Eigenproblem, 1992
// (4) Lapack dlaed4.f

// compile if type dependent public function names defined
#if defined(__ARMAS_PROVIDES) && defined(__ARMAS_REQUIRES)
// ------------------------------------------------------------------------------

//! \cond
#include "internal.h"
#include "matrix.h"
#include "internal_lapack.h"
//! \endcond

/*
 * Compute value of rational function sum z(j)^2/delta(j) and its derivative
 * sum (z(j)/delta(j))^2 forwards in range [start, end). 
 */
static inline
void rational_forward(DTYPE *val, DTYPE *dval, 
                      armas_x_dense_t *Z, armas_x_dense_t *delta, int start, int end)
{
    int i;
    DTYPE dj, zj, tval;

    for (i = start; i < end; i++) {
        dj = armas_x_get_at_unsafe(delta, i);
        zj = armas_x_get_at_unsafe(Z, i);
        tval = zj/dj;
        *val += zj*tval;
        *dval += tval*tval;
    }
}

/*
 * Compute value of rational function sum z(j)^2/delta(j) and its derivative
 * sum (z(j)/delta(j))^2 backwards in range [start, end). 
 */
static inline
void rational_backward(DTYPE *val, DTYPE *dval, 
                       armas_x_dense_t *Z, armas_x_dense_t *delta, int start, int end)
{
    int i;
    DTYPE dj, zj, tval;

    for (i = end-1; i >= start; i--) {
        dj = armas_x_get_at_unsafe(delta, i);
        zj = armas_x_get_at_unsafe(Z, i);
        tval = zj/dj;
        *val += zj*tval;
        *dval += tval*tval;
    }
}

/*
 * Compute delta(j) = (D(j) - D(K)) - tau
 */
static inline
void compute_delta(armas_x_dense_t *delta, armas_x_dense_t *D, int ik, DTYPE tau) {
    int i, nx, kx, n, k;
    DTYPE d0, *dp, *dk;
    dp = armas_x_data(delta);
    dk = armas_x_data(D);
    nx = delta->rows == 1 ? delta->step : 1;
    kx = D->rows == 1 ? D->step : 1;
    d0 = armas_x_get_at_unsafe(D, ik);
    for (i = 0, n = 0, k = 0; i < armas_x_size(D); i++, n += nx, k += kx) {
        dp[n] = dk[k] - d0 - tau;
    }
}

/*
 * Compute delta(j) = delta(j) - eta
 */
static inline
void update_delta(armas_x_dense_t *delta, DTYPE eta) {
    int i, ix, n;
    DTYPE *dp;
    dp = armas_x_data(delta);
    ix = delta->rows == 1 ? delta->step : 1;
    for (i = 0, n = 0; i < armas_x_size(delta); i++, n += ix) {
        dp[n] = dp[n] - eta;
    }
}


/*
 * Secular function
 *      f(x) = 1/r + sum k; z_k^2/(d_k - x)  k = 1 .. n
 *
 * approximated with rational function
 *      q(x) = s + c1/(d_k - x) + c2/(d_k1 - x)
 *
 * set q(x) = 0 and solve for roots in [d(k), d(k+1)]
 *
 *      s(d_k - x)(d_k1 - x) + c1*(d_k1 - x) + c2*(dk - x) = 0
 *     
 * assume approximation y of x, x = y + w and write 
 *
 *      s*(dk - y - w)(d_k1 - y - w) + c1*(d_k1 - y - w) + c2*(d_k - y - w) =
 *      s*(D_k - w)(D_k1 - w) + c1*(D_k1 - w) + c2*(D_k - w)
 *      s*(D_k*D_k1 - (D_k + D_k1)*w + w^2) + c1*(D_k1 - w) + c2*(D_k - w)
 *      s*w^2 - (s*(D_k + D_k1) + c1 + c2)*w - s*D_k*D_k1 + c1*D_k1 + c2*D_k
 *
 *      a = s
 *      b = s*(D_k + D_k1) + c1 + c2
 *      c = s*D_k*D_k1 + c1*D_k1 + c2*D_k
 *      
 * where D_k = d_k - y and D_k1 = d_k1 - y
 *
 * to compute coefficient s, c1, c2 write
 *
 *      f(y) = 1/r + H(y) + G(y)  
 * where
 *      H(y) = sum k; z_k^2/(d_k - y)  k = 1 .. i   (H(y) < 0 as d_k <= y)
 *      G(y) = sum k; z_k^2/(d_k - y)  k = i+1 .. n (G(y) > 0 as d_k >= y)
 *      
 * set
 *      h(y) = s1 + c1/(d_k - y), require h(y) == H(y) and h'(y) == H'(y)
 * and solve for s1, c1
 *      c1 = (d_k - y)^2 * H'(y)    = D_k^2*H'(y)
 *      s1 = H(y) - (d_k - y)*H'(y) = H(y) - D_k*H'(y)
 *
 * set
 *      g(y) = s2 + c2/(d_k1 - y), require g(y) == G(y) and g'(y) == G'(y)
 * and solve for s2, c2
 *      c2 = (d_k1 - y)^2 * G'(y)    = D_k1^2*G'(y)
 *      s2 = G(y) - (d_k1 - y)*G'(y) = G(y) - D_k1*G'(y)
 *      
 * have
 *      f(y) = 1/r + s1 + s2 + c1/(d_k - y) + c2/(d_k1 - y)
 *
 *      s = 1/r + s1 + s2 = 1/r + H(y) - D_k*H'(y) + G(y) - D_k1*G'(y)
 *        = 1/r + H(y) + G(y) - D_k*H'(y) - D_k1*G'(y)
 *        = F(y) - D_k*H'(y) - D_k1*G'(y)
 *        
 * coefficents for quadratic equation a*w^2 - b*w + c
 *
 *      a = F(y) - D_k*H'(y) - D_k1*G'(y)
 *      b = (D_k + D_k1)*(F(y) - D_k*H'(y) - D_k1*G'(y))  - Dk1^2*G'(y) - Dk^2*H'(y)
 *        = (D_k + D_k1)*F(y) - D_k*D_k1*H'(y) - D_k*D_k1*G'(y) 
 *        = (D_k + D_k1)*F(y) - D_k*D_k1*(H'(y) + G'(y))
 *      c = D_k*D_k1*(F(y) - D_k*H'(y) - D_k1*G'(y)) + D_k1*D_k^2*H'(y) + D_k*D_k1^2*G'(y)
 *        = D_k*D_k1*F(y)
 *        
 * solve for w
 * 
 *      w = (b - sqrt(b^2 - 4*a*c))/2*a  if b <= 0
 *        = 2*c/(b + sqrt(b^2 - 4*a*c)   if b >  0
 *
 * w is next increment to current approximation y.
 *
 * Initial guess at y0 = (d_k + d_k1)/2
 * 
 * if F(y0) > 0 then lambda closer to d_k, select d_k as origin. And if F(y0) < 0
 * then lambda closer to d_k1 and select it as origin.
 *
 */

/*
 * Stopping criterias:
 * a.  |eta| <= c *__eps * min(|d_k - y|, |d_k1 - y|)               (1) eq.48
 * b.  eta^2 <= __eps * min(|d_k - y|, |d_k1 - y|)*(|eta0|-|eta|)   (1) eq.49
 * c.  |F|   <= N * __eps * (1/rho + |H+G|)                         (1) eq.50/(3) eq 3.4
 */

/*
 * \brief Guess initial value for iteration
 */
static
int __trdsec_initial_guess(DTYPE *tau, DTYPE *tau_low, DTYPE *tau_high, 
                           armas_x_dense_t *D, armas_x_dense_t *Z,
                           armas_x_dense_t *delta, int index, DTYPE rho)
{
    DTYPE G, Hx, dG, F, A, B, C, dd;
    DTYPE d_k, d_k1, z_k, z_k1, mpoint, diff; 
    int N, iK, iN, last = 0;

    N = armas_x_size(D);
    if (index == N-1) {
        iN = N-2;
        last = 1;
    } else {
        iN = index;
    }

    // compute initial value at (D[k] + D[k+1])/2
    // note: D[i] - (D[k] + D[k+1])/2 == D[i] - D[k] - (D[k+1] - D[k])/2.0
    d_k  = armas_x_get_at_unsafe(D, iN);
    d_k1 = armas_x_get_at_unsafe(D, iN+1);
    diff = d_k1 - d_k;
    mpoint  = last ? rho/2.0 : diff/2.0;

    // computes: delta[i] = D[i] - D[index] - midpoint
    compute_delta(delta, D, index, mpoint);
    
    // initial guess as (1) section 4, equations (40 - 44)
    // H = 1/rho + sum j; z_j^2/(d_j - y), j = 0..N-1, j != index, index+1
    G   = __ZERO; dG = __ZERO;
    rational_forward(&G, &dG, Z, delta, 0, iN+1);
    rational_backward(&G, &dG, Z, delta, iN+1, N);
    // G = z_k^2/(d_k - y) + z_k1^2/(d_k1 - y)
    z_k  = armas_x_get_at_unsafe(Z, iN);
    z_k1 = armas_x_get_at_unsafe(Z, iN+1);
    d_k  = armas_x_get_at_unsafe(delta, iN);
    d_k1 = armas_x_get_at_unsafe(delta, iN+1);

    // F is f(x) at initial point, 1/rho + g(y) + h(y)
    F  = 1/rho + G;
    // Hx is value of h(y) at initial point
    Hx   = z_k*(z_k/d_k) + z_k1*(z_k1/d_k1);
    // C is g(x) at initial point
    C  = F - Hx;
    if (last)
        goto lastentry;

    // F is value of secular function at initial point, if F > 0 then
    // lambda is closer to D[k], otherwise closer to D[k+1]
    if (F > 0) {
        // (1) theorem.6: d_k < d_k + tau < lambda_k < (d_k + d_k1)/2
        iK = index;
        A  = z_k*z_k*diff;
        B  = C*diff + z_k*z_k + z_k1*z_k1;
        *tau_low = __ZERO;
        *tau_high = mpoint;
    } else {
        // (1) theorem.6: (d_k + d_k1)/2 < lamdba_k < d_k1 + tau < d_k1
        iK = index + 1;
        A  = -z_k1*z_k1*diff;
        B  = -C*diff + z_k*z_k + z_k1*z_k1;
        *tau_low = -mpoint;
        *tau_high = __ZERO;
    }
    B /= 2.0;
    armas_x_discriminant(&dd, A, B, C);
    if (B > __ZERO) {
        *tau = A/(B + __SQRT(dd));
    } else {
        *tau = (B - __SQRT(dd))/C;
    }
    compute_delta(delta, D, iK, *tau);
    return iK;

 lastentry:
    // Compute here initial guess when index == N-1, ie. last element

    A  = -z_k1*z_k1*diff;
    B  = -C*diff + z_k*z_k + z_k1*z_k1;
    B  /= 2.0;
    Hx = z_k*z_k/(diff + rho) + z_k1*z_k1/rho;
    if (F <= __ZERO && C <= Hx) {
        *tau = rho;
    } else {
        armas_x_discriminant(&dd, A, -B, C);
        if (B < __ZERO) {
            *tau = A/(B - __SQRT(dd));
        } else {
            *tau = (B + __SQRT(dd))/C;
        }
    }
    if (F < __ZERO) {
        *tau_low = mpoint;
        *tau_high = rho;
    } else {
        *tau_low = __ZERO;
        *tau_high = mpoint;
    }
    compute_delta(delta, D, N-1, *tau);
    return index-1;
}

/*
 * \brief Compute i'th root of secular function
 *
 * Compute root of secular function by rational approximation.
 *
 * \param[out] lambda
 *      Computed root.
 * \param[in] D
 *      Diagonal vector
 * \param[in] Z
 *      Rank-one update vector
 * \param[out] delta
 *      On exit, D - lamdba*I
 * \param[in] index
 *      Number of requested root
 * \param[in] rho
 *      Coefficient
 */
static
int __trdsec_root(DTYPE *lambda, armas_x_dense_t *D, armas_x_dense_t *Z,
                  armas_x_dense_t *delta, int index, DTYPE rho)
{
    int iK, iK1, niter, maxiter, N;

    DTYPE H, dH, G, dG, F, dF, Fa, A, B, C, tau, tau_low, tau_high, eta, eta0, dd, edif;
    DTYPE delta_k, delta_k1, da_k, da_k1; 

    N    = armas_x_size(D);
    tau  = __ZERO;

    // compute initial value at (D[k] + D[k+1])/2
    // note: D[i] - (D[k] + D[k+1])/2 == D[i] - D[k] - (D[k+1] - D[k])/2.0
    iK = __trdsec_initial_guess(&tau, &tau_low, &tau_high, D, Z, delta, index, rho);
    if (iK == index) {
        iK1 = index+1;
        delta_k  = armas_x_get_at_unsafe(delta, iK);
        if (index < N-1) {
            delta_k1 = armas_x_get_at_unsafe(delta, iK1);
        } else {
            delta_k1 = tau;
        }
    } else {
        iK1 = index;
        delta_k1  = armas_x_get_at_unsafe(delta, iK1);
        if (index < N-1) {
            delta_k = armas_x_get_at_unsafe(delta, iK);
        } else {
            delta_k = tau;
        }
    }

    eta  = __ONE;
    eta0 = __ONE;
    maxiter = 20;
    for (niter = 0; niter < maxiter; niter++) {

        G    = __ZERO; dG = __ZERO;
        rational_forward(&G, &dG, Z, delta, 0, iK+1);
        H    = __ZERO; dH = __ZERO;
        rational_backward(&H, &dH, Z, delta, iK+1, N);
        F  = 1/rho + G + H;
        dF = dG + dH;
        Fa = 1/rho + __ABS(G+H);

        da_k  = __ABS(delta_k);
        da_k1 = __ABS(delta_k1);

        edif  = (da_k < da_k1 ? da_k : da_k1)*(__ABS(eta0) - __ABS(eta));

        // stopping criterion from (1) eq 50 and (3) eq 3.4
        if (__ABS(F) < N*__EPS*Fa) {
            break;
        }
        // stopping criterion from (1), eq 49
        if (eta*eta < __EPS*edif) {
            break;
        }
        if (F < __ZERO) {
            tau_low = tau_low > tau ? tau_low : tau;
        } else {
            tau_high = tau_high < tau ? tau_high : tau;
        }

        A = F - delta_k*dG - delta_k1*dH;
        B = (delta_k + delta_k1)*F - delta_k*delta_k1*(dH + dG);
        C = delta_k*delta_k1*F;
        // compute discriminant of A*x^2 - 2B*x + C with extra precission
        B /= 2.0;
        armas_x_discriminant(&dd, A, B, C);
        eta0 = eta;
        if (B > 0) {
            eta = C/(B + __SQRT(dd));
        } else {
            eta = (B - __SQRT(dd))/A;
        }
        // F and eta should be of different sign. If product positive
        // then use Newton step instead.
        if (F*eta > __ZERO) {
            eta = -F/dF;
        }
        // If overshooting, adjust ...
        if (tau+eta > tau_high || tau+eta < tau_low) {
            if (F < __ZERO) {
                eta = (tau_high - tau)/2.0;
            } else {
                eta = (tau_low - tau)/2.0;
            }
        }

        tau += eta;
        delta_k  -= eta;
        delta_k1 -= eta;
        update_delta(delta, eta);
    }

    if (index == N-1) {
        *lambda = armas_x_get_at_unsafe(D, N-1) + tau;
    } else {
        *lambda = armas_x_get_at_unsafe(D, iK) + tau;
    }
    return niter == maxiter ? -niter : niter;
}

/*
 * \brief 
 */
static
void __update_vec_delta(DTYPE *vk, armas_x_dense_t *d,
                        armas_x_dense_t *delta, int index, DTYPE rho)
{
    DTYPE n0, n1, dn, dk, val, p0, p1;
    int j, N;

    N  = armas_x_size(d);
    dk = armas_x_get_at_unsafe(d, index);
    dn = armas_x_get_at_unsafe(delta, N-1);

    // compute; prod j; (lambda_j - d_k)/(d_j - d_k) , j = 0..index-1
    p0 = __ONE;
    for (j = 0; j < index; j++) {
        n0 = armas_x_get_at_unsafe(delta, j);
        n1 = armas_x_get_at_unsafe(d, j) - dk;
        p0 *= n0/n1;
    }
    p1 = __ONE;
    // compute; prod j; (lambda_j - d_k)/(d_j+1 - d_k) , j = index..N-2
    for (j = index; j < N-1; j++) {
        n0 = armas_x_get_at_unsafe(delta, j);
        n1 = armas_x_get_at_unsafe(d, j+1) - dk;
        p1 *= n0/n1;
    }

    val = p0*p1*(dn/rho);
    *vk = __SQRT(__ABS(val));
}


/*
 * \brief Compute updated rank-1 update vector with precomputed deltas
 *
 * \param[out] z
 *      On exit, the updated rank-1 vector
 * \param[in] Q
 *      Precomputed deltas
 * \param[in] d
 *      Original parameters of secular function
 * \param[in] rho
 *      Coefficient
 *
 * For details see (3). Computation as equation 3.3 in (3)
 */
static inline
void __trdsec_update_vec_delta(armas_x_dense_t *z, armas_x_dense_t *Q,
                               armas_x_dense_t *d, DTYPE rho)
{
    armas_x_dense_t delta;
    DTYPE zk;
    int i;
    EMPTY(delta);

    for (i = 0; i < armas_x_size(z); i++) {
        armas_x_column(&delta, Q, i);
        __update_vec_delta(&zk, d, &delta, i, rho);
        armas_x_set_at_unsafe(z, i, zk);
    }
}

/*
 * \brief Compute eigenvector corresponding precomputed deltas (d_j - lambda)
 *
 *   qi = ( z_1/(d_1 - l), ..., z_n/(d_n - l)) / sqrt(sum j; z_j^2/(d_j - l)^2)
 *
 *  See (3) equation 3.1.
 *
 *  NOTE: This version works if the provided deltas are computed to high precision
 *  when solving the secular function.
 */
static inline
void __trdsec_eigenvec_delta(armas_x_dense_t *qi, armas_x_dense_t *dl, armas_x_dense_t *z)
{
    DTYPE dk, zk, t, s;
    int k;
    for (k = 0; k < armas_x_size(dl); k++) {
        zk = armas_x_get_at_unsafe(z, k);
        dk = armas_x_get_at_unsafe(dl, k);
        t = zk/dk;
        armas_x_set_at_unsafe(qi, k, t);
    }
    s = armas_x_nrm2(qi, (armas_conf_t *)0);
    armas_x_invscale(qi, s, (armas_conf_t *)0);
}

/*
 * \brief Compute eigenvectors using precomputed deltas
 *
 * \param[out] Q
 *      On exit the column eigenvectors
 * \param[in] z
 *      Rank-1 update vector
 * \param[in] Q2
 *      Matrix of row vectors that hold precomputed deltas (dk - l_k) for all
 *      eigenvalues. (output from __trdsec_solve2()).
 *
 * This version requires space 2*N^2 (target eigenmatrix Q and workspace for deltas).
 */
static
void __trdsec_eigen2_build(armas_x_dense_t *Q, armas_x_dense_t *z, armas_x_dense_t *Q2)
{
    armas_x_dense_t qi, delta;
    int k;
    EMPTY(delta);

    for (k = 0; k < armas_x_size(z); k++) {
        armas_x_column(&qi, Q, k);
        armas_x_row(&delta, Q2, k);
        __trdsec_eigenvec_delta(&qi, &delta, z);
    }
}

/*
 * \brief Compute eigenvectors using precomputed deltas
 *
 * \param[in,out] Q
 *      On entry the row vectors of precomputed deltas. On exit the column eigenvectors
 * \param[in] z
 *      Rank-1 update vector
 *
 *  This function computes updated eigenvectors in-place. It assumes precomputed
 *  deltas are row vectors of Q and result eigenvectors are column vectors of Q.
 */
static
void __trdsec_eigenbuild_inplace(armas_x_dense_t *Q, armas_x_dense_t *z)
{
    DTYPE zk0, zk1, dk0, dk1, t;
    int k, i;
    armas_x_dense_t QTL, QBR, Q00, q11, q12, q21, Q22, qi;

    EMPTY(q11); EMPTY(q12); EMPTY(q21); EMPTY(Q00);
    
    __partition_2x2(&QTL,  __nil,
                    __nil, &QBR,  /**/ Q, 0, 0, ARMAS_PTOPLEFT);
    while (QBR.rows > 0) {
        __repartition_2x2to3x3(&QTL,
                               &Q00,  __nil, __nil,
                               __nil, &q11,  &q12,
                               __nil, &q21,  &Q22,  /**/ Q, 1, ARMAS_PBOTTOMRIGHT);
        //----------------------------------------------------------------------------
        k   = Q00.rows;
        zk0 = armas_x_get_at_unsafe(z, k);
        dk0 = armas_x_get_unsafe(&q11, 0, 0);
        armas_x_set_unsafe(&q11, 0, 0, zk0/dk0);

        for (i = 0; i < armas_x_size(&q12); i++) {
            zk1 = armas_x_get_at_unsafe(z, k+i+1);
            dk0 = armas_x_get_at_unsafe(&q12, i);
            dk1 = armas_x_get_at_unsafe(&q21, i);
            armas_x_set_at_unsafe(&q12, i, zk0/dk1);
            armas_x_set_at_unsafe(&q21, i, zk1/dk0);
        }
        //----------------------------------------------------------------------------
        __continue_3x3to2x2(&QTL, __nil,
                            __nil, &QBR, /**/ &Q00, &q11, &Q22, Q, ARMAS_PBOTTOMRIGHT);
    }
    // scale column eigenvector
    for (k = 0; k < armas_x_size(z); k++) {
        armas_x_column(&qi, Q, k);
        t = armas_x_nrm2(&qi, (armas_conf_t *)0);
        armas_x_invscale(&qi, t, (armas_conf_t *)0);
    }
}

// ---------------------------------------------------------------------------------
// PUBLIC FUNCTIONS

/**
 * \brief Solve secular function \f$ 1 + rho * sum_k { z_k^2/(z_k - y) } \f$
 *
 * \param[out] y
 *      On exit, roots of secular function
 * \param[in] d, z
 *      Parameters of secular function
 * \param[in] delta
 *      Workspace
 * \param[in] rho
 *      Coefficient
 * \param[in] conf 
 *      Connfiguration block
 *
 * \retval  0 Success
 * \retval <0 Error, `conf.error` holds error code and return value index of first 
 *     non-convergent root.
 * \ingroup lapack
 */
int armas_x_trdsec_solve(armas_x_dense_t *y, armas_x_dense_t *d,
                         armas_x_dense_t *z, armas_x_dense_t *delta, DTYPE rho,
                         armas_conf_t *conf)
{
    DTYPE dlam;
    int i, err = 0;

    if (!conf)
        conf = armas_conf_default();

    // sizes need to match
    if (armas_x_size(d) != armas_x_size(y)
        || armas_x_size(d) != armas_x_size(delta)
        || armas_x_size(d) != armas_x_size(z)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }

    for (i = 0; i < armas_x_size(d); i++) {
        if (__trdsec_root(&dlam, d, z, delta, i, rho) < 0 && err == 0) {
            err = -(i+1);
        }
        armas_x_set_at_unsafe(y, i, dlam);
    }
    if (err)
        conf->error = ARMAS_ECONVERGE;
    return err;
}


/**
 * \brief Solve secular function and compute rank-1 update vector.
 *
 * \param[out] y
 *      On exit the updated eigenvalues
 * \param[out] v
 *      On exit the updated rank-1 vector
 * \param[out] Qd
 *      On exit row vectors of delta values computed by secular function root finder.
 * \param[in] d
 *      Original eigenvalues
 * \param[in] z
 *      Original rank-1 vector
 * \param[in] rho
 *      Coefficient
 * \param[in] conf
 *      Optional configuration block
 * \ingroup lapack
 */
int armas_x_trdsec_solve_vec(armas_x_dense_t *y, armas_x_dense_t *v, armas_x_dense_t *Qd,
                             armas_x_dense_t *d, armas_x_dense_t *z,
                             DTYPE rho, armas_conf_t *conf)
{
    armas_x_dense_t delta;
    DTYPE dlam;
    int i, err = 0;
    if (!conf)
        conf = armas_conf_default();

    if (armas_x_size(v) != armas_x_size(z)
        || armas_x_size(v) != armas_x_size(d)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    for (i = 0; i < armas_x_size(d); i++) {
        armas_x_row(&delta, Qd, i);
        if (__trdsec_root(&dlam, d, z, &delta, i, rho) < 0 && err == 0) {
            err = -(i+1);
        }
        armas_x_set_at_unsafe(y, i, dlam);
    }
    if (err == 0) {
        __trdsec_update_vec_delta(v, Qd, d, rho);
    } else {
        conf->error = ARMAS_ECONVERGE;
    }
    return err;
}

/**
 * \brief Compute the eigenvectors corresponding the updated eigenvalues.
 *
 * \param[in,out] Q
 *      On entry if Qd is null, the precomputed deltas. On exit updated eigenvectors
 * \param[in] v
 *      Updated rank-one update vector
 * \param[in] Qd
 *      Precomputed deltas as returned by __trdsec_solve_vec().
 * \param[in,out] conf
 *      Configuration block
 *
 *  If Qd is null or same matrix as Q then eigenvectors are computed in-place. 
 *
 * \retval  0 Success
 * \retval -1 Error, `conf.error` holds error code.
 * \ingroup lapack
 */
int armas_x_trdsec_eigen(armas_x_dense_t *Q, armas_x_dense_t *v, armas_x_dense_t *Qd,
                         armas_conf_t *conf)
{
    if (!conf)
        conf = armas_conf_default();

    if (Q->rows != Q->cols || Qd->rows != Qd->cols) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
    if (Q->rows != armas_x_size(v) || Qd->rows != armas_x_size(v)) {
        conf->error = ARMAS_ESIZE;
        return -1;
    }
        
    if (!Qd || Q == Qd) {
        __trdsec_eigenbuild_inplace(Q, v);
    } else {
        __trdsec_eigen2_build(Q, v, Qd);
    }
    return 0;
}


#if defined(__BROKEN__)
// following are included here for time being as examples. These break down
// if new eigenvalue is close to original and difference between them goes to zero.

/*
 * \brief Compute updated element of rank-1 update vector 
 *
 *     p0 = prod [j=0,k-1]; (d_k - dl_j)/(d_k - d_j) 
 *     p1 = prod [j=k,n-1]; (dl_j - d_k)/(d_j+1 - d_k)
 *     v_k = sqrt( p0*p1*(dl_n - d_k)/rho )
 *
 *  Computes as in (3) equation 3.3
 */
static inline
void __update_vec(DTYPE *vk, armas_x_dense_t *d, armas_x_dense_t *dl, int index, DTYPE rho)
{
    DTYPE n0, n1, dn, dk, val, p0, p1;
    int k, N;

    N  = armas_x_size(d);
    dk = armas_x_get_at_unsafe(d, index);
    dn = armas_x_get_at_unsafe(dl, N-1);

    // compute; prod j; (lambda_j - d_k)/(d_j - d_k) , j = 0..index-1
    p0 = __ONE;
    for (k = 0; k < index; k++) {
        n0 = dk - armas_x_get_at_unsafe(dl, k);
        n1 = dk - armas_x_get_at_unsafe(d, k);
        p0 *= n0/n1;
    }
    p1 = __ONE;
    // compute; prod j; (lambda_j - d_k)/(d_j+1 - d_k) , j = index..N-2
    for (k = index; k < N-1; k++) {
        n0 = armas_x_get_at_unsafe(dl, k) - dk;
        n1 = armas_x_get_at_unsafe(d, k+1) - dk;
        p1 *= n0/n1;
    }

    val = p0*p1*((dn - dk)/rho);
    *vk = __SQRT(__ABS(val));
}



/*
 * \brief Compute updated rank-1 update vector with updated eigenvalues
 *
 * \param[out] z
 *      On exit, the updated rank-1 vector
 * \param[in] dl
 *      Roots of secular function as returned by __secular_solve()
 * \param[in] d
 *      Original parameters of secular function
 * \param[in] rho
 *      Coefficient
 *
 * For details see (3). Computation as equation 3.3 in (3)
 */
static inline
void __trdsec_update_vec(armas_x_dense_t *z, armas_x_dense_t *dl,
                         armas_x_dense_t *d, DTYPE rho)
{
    DTYPE zk;
    int i;
    for (i = 0; i < armas_x_size(d); i++) {
        __update_vec(&zk, d, dl, i, rho);
        armas_x_set_at_unsafe(z, i, zk);
    }
}


/*
 * \brief Compute eigenvector corresponding eigenvalue lmbda
 *
 *   qi = ( z_1/(d_1 - l), ..., z_n/(d_n - l)) / sqrt(sum j; z_j^2/(d_j - l)^2)
 *
 *  See (3) equation 3.1.
 */
static inline
void __trdsec_eigenvec(armas_x_dense_t *qi, armas_x_dense_t *d,
                       armas_x_dense_t *z, DTYPE lmbda)
{
    DTYPE dk, zk, t, s;
    int k;
    for (k = 0; k < armas_x_size(d); k++) {
        zk = armas_x_get_at_unsafe(z, k);
        dk = armas_x_get_at_unsafe(d, k);
        t = zk/(dk - lmbda);
        armas_x_set_at_unsafe(qi, k, t);
    }
    s = armas_x_nrm2(qi, (armas_conf_t *)0);
    armas_x_invscale(qi, s, (armas_conf_t *)0);
}

/*
 * \brief Compute eigenvectors 
 */
static
void __trdsec_eigen_build(armas_x_dense_t *Q, armas_x_dense_t *dl,
                          armas_x_dense_t *d, armas_x_dense_t *z)
{
    armas_x_dense_t qi;
    DTYPE lmbda;
    int k;
    for (k = 0; k < armas_x_size(d); k++) {
        armas_x_column(&qi, Q, k);
        lmbda = armas_x_get_at_unsafe(dl, k);
        __trdsec_eigenvec(&qi, d, z, lmbda);
    }
}
#endif

#endif /* __ARMAS_PROVIDES && __ARMAS_REQUIRES */

// Local Variables:
// c-basic-offset: 4
// indent-tabs-mode: nil
// End:

