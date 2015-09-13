
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "eft.h"

#define OK(exp) ((exp) ? 'Y' : 'N')


int test_twosum_f32(int verbose)
{
  int nfails = 0;
  float x, y;
  float a = 1e3;
  float b = 1.0 + FLT_EPSILON;

  if (verbose)
    printf("twosum_f32: |a| > |b|, 1000.0 + (1.0 + eps)\n");

  twosum_f32(&x, &y, a, b);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\t     a+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  twosum_f32(&x, &y, b, a);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\t     b+a, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  x = a;
  twosum_f32(&x, &y, x, b);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\tx=a; x+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  x = b;
  twosum_f32(&x, &y, a, x);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\tx=b; a+x, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));
  return nfails;

  y = a;
  twosum_f32(&x, &y, y, b);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\ty=a; y+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  y = b;
  twosum_f32(&x, &y, a, y);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\ty=b; a+y, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));
  return nfails;

}

int test_twosum_extf32(int verbose)
{
  int nfails = 0;
  float x, y;
  float ah = 1e3;
  float al = -FLT_EPSILON;
  float bh = 1.0 + FLT_EPSILON;
  float bl = 0.0;

  if (verbose)
    printf("twosum_extf32: |ah,al| > |bh,bl|, (1000.0, -eps) + (1.0+eps, 0.0)\n");

  twosum_f32_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\t     (ah,al)+(bh,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  twosum_f32_ext(&x, &y, bh, bl, ah, al);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\t     (bh,bl)+(ah,al), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  x = ah;
  twosum_f32_ext(&x, &y, x, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\tx=ah; (x,al)+(bh,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  x = bh;
  twosum_f32_ext(&x, &y, ah, al, x, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\tx=bh; (ah,al)+(x,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  y = ah;
  twosum_f32_ext(&x, &y, y, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\ty=ah; (y,al)+(bh,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  y = bh;
  twosum_f32_ext(&x, &y, ah, al, y, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\ty=bh; (ah,al)+(y,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));
  return nfails;
}

int test_fastsum_f32(int verbose)
{
  int nfails = 0;
  float x, y;
  float a = 1e3;
  float b = 1.0 + FLT_EPSILON;

  if (verbose)
    printf("fastsum_f32: |a| > |b|, 1000.0 + (1.0 + eps)\n");

  fastsum_f32(&x, &y, a, b);
  nfails += 1 - (int)(x == 1001.0 && y == FLT_EPSILON);
  if (verbose)
    printf("\t     a+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  x = a;
  fastsum_f32(&x, &y, x, b);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\tx=a; x+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  x = b;
  fastsum_f32(&x, &y, a, x);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\tx=b; a+x, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));
  return nfails;

  y = a;
  fastsum_f32(&x, &y, y, b);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\ty=a; y+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));

  y = b;
  fastsum_f32(&x, &y, a, y);
  nfails += 1 - (int)(x==1001.0 && y==FLT_EPSILON);
  if (verbose)
    printf("\ty=b; a+y, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==FLT_EPSILON));
  return nfails;
}

int test_fastsum_extf32(int verbose)
{
  int nfails = 0;
  float x, y;
  float ah = 1e3;
  float al = -FLT_EPSILON;
  float bh = 1.0 + FLT_EPSILON;
  float bl = 0.0;

  if (verbose)
    printf("fastsum_extf32: (1000.0, -eps) + (1.0+eps, 0.0)\n");

  fastsum_f32_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x == 1001.0 && y == 0.0);

  if (verbose)
    printf("\t|a|>|b|, x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));
  return nfails;
}

int test_twoprod_f32(int verbose)
{
  int nfails = 0;
  float x, y;
  float a = 2.0 - 2*FLT_EPSILON;
  float b = 2.0 + 2*FLT_EPSILON;
  const float x_exp = 4.0;
  const float y_exp = -4.0*FLT_EPSILON*FLT_EPSILON;

  if (verbose)
    printf("twoprod_f32: (1.0 - eps) * (1.0 + eps)\n");

  twoprod_f32(&x, &y, a, b);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\t     a*b; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));

  twoprod_f32(&x, &y, b, a);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\t     b*a; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));

  x = a;
  twoprod_f32(&x, &y, x, b);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\tx=a; x*b; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));

  x = b;
  twoprod_f32(&x, &y, a, x);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\tx=b; a*x; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));

  y = a;
  twoprod_f32(&x, &y, y, b);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\ty=a; y*b; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));

  y = b;
  twoprod_f32(&x, &y, a, y);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\ty=b; a*x; x==%5.1f && y==%e  : '%c'\n", x_exp, y_exp, OK(x==x_exp && y==y_exp));
  return nfails;
}

int test_twoprod_extf32(int verbose)
{
  int nfails = 0;
  float x, y;
  float ah = 2.0 - 2.0*FLT_EPSILON;
  float al = FLT_EPSILON*FLT_EPSILON;
  float bh = 2.0 + 2.0*FLT_EPSILON;
  float bl = FLT_EPSILON*FLT_EPSILON;
  const float x_exp = 4.0;
  const float y_exp = 0.0;

  if (verbose)
    printf("twoprod_extf32: (2.0 - 2.0*eps, eps*eps) * (2.0 + 2.0*eps, eps*eps)\n");

  twoprod_f32_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\t      a*b; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  twoprod_f32_ext(&x, &y, bh, bl, ah, al);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\t      b*a; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  x = ah;
  twoprod_f32_ext(&x, &y, x, al, bh, bl);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\tx=ah; a*b; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  x = bh;
  twoprod_f32_ext(&x, &y, ah, al, x, bl);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\tx=bh; b*a; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  y = ah;
  twoprod_f32_ext(&x, &y, y, al, bh, bl);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\ty=ah; a*b; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  y = bh;
  twoprod_f32_ext(&x, &y, ah, al, y, bl);
  nfails += 1 - (int)(x==x_exp && y==y_exp);
  if (verbose)
    printf("\ty=bh; b*a; x==1.0 && y==0.0  : '%c'\n", OK(x==x_exp && y==y_exp));

  return nfails;
}

int test_twodiv_f32(int verbose)
{
  int y_ok, x_ok, nfails = 0;
  float x, y;
  float b = 33.0;
  float a = 1.0 - FLT_EPSILON;

  if (verbose)
    printf("twodiv_f32: (1.0 - eps)/33.0\n");

  // approximate error term is |a/b - x| <= eps*|a|
  //     ==> |y| <= eps*|a| && |a-b*x| <= eps*|a|
  twodiv_f32(&x, &y, a, b);

  x_ok = fabsf(a-b*x) <= FLT_EPSILON*fabsf(a);
  y_ok = fabsf(y)     <= FLT_EPSILON*fabsf(a);

  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a-b*x| <= eps*|a| && |y| <= eps*|a|  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}

int test_split_f32(int verbose)
{
  int ok, nfails = 0;
  float x, y;
  float a = 2.0*__factor_f32 + __factor_f32/2.0;

  if (verbose)
    printf("split_f32: 2*fact32 + fact32/2\n");

  split_f32(&x, &y, a);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\t     x+y = a  : '%c'\n", OK(ok));

  x = a;
  split_f32(&x, &y, x);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\tx=a; x+y = x  : '%c'\n", OK(ok));

  y = a;
  split_f32(&x, &y, y);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\ty=a; x+y = y  : '%c'\n", OK(ok));
  return nfails;
}


int test_twosum_f64(int verbose)
{
  int nfails = 0;
  double x, y;
  double a = 1e3;
  double b = 1.0 + DBL_EPSILON;

  if (verbose)
    printf("twosum_f64: |a| > |b|,  1000.0 + (1.0 + eps)\n");


  twosum_f64(&x, &y, a, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);

  if (verbose)
    printf("\t     a+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  twosum_f64(&x, &y, b, a);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\t     b+a, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  x = a;
  twosum_f64(&x, &y, x, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\tx=a; x+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  x = b;
  twosum_f64(&x, &y, a, x);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\tx=b; a+x, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));
  return nfails;

  y = a;
  twosum_f64(&x, &y, y, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\ty=a; y+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  y = b;
  twosum_f64(&x, &y, a, y);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\ty=b; a+y, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  return nfails;
}

int test_twosum_extf64(int verbose)
{
  int nfails = 0;
  double x, y;
  double ah = 1e3;
  double al = -DBL_EPSILON;
  double bh = 1.0 + DBL_EPSILON;
  double bl = 0.0;

  if (verbose)
    printf("twosum_extf64: |a| > |b|  (1000.0, -eps) + (1.0+eps, 0.0)\n");

  twosum_f64_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\t      (ah,al)+(bh,bl), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  twosum_f64_ext(&x, &y, bh, bl, ah, al);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\t      (bh,bl)+(ah,al), x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  x = ah;
  twosum_f64_ext(&x, &y, x, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\tx=ah; (x,al)+(bh,bl),  x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  x = bh;
  twosum_f64_ext(&x, &y, ah, al, x, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\tx=bh; (ah,al)+(x,bl),  x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  y = ah;
  twosum_f64_ext(&x, &y, y, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\ty=ah; (y,al)+(bh,bl),  x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));

  y = bh;
  twosum_f64_ext(&x, &y, ah, al, y, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\ty=bh; (ah,al)+(y,bl),  x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));
  return nfails;
}

int test_fastsum_f64(int verbose)
{
  int nfails = 0;
  double x, y;
  double a = 1e3;
  double b = 1.0 + DBL_EPSILON;

  if (verbose)
    printf("fastsum_f64: |a| > |b|,  1000.0 + (1.0 + eps)\n");

  fastsum_f64(&x, &y, a, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\t     a+b,  x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  x = a;
  fastsum_f64(&x, &y, x, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\tx=a; x+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  x = b;
  fastsum_f64(&x, &y, a, x);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\tx=b; a+x, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));
  return nfails;

  y = a;
  fastsum_f64(&x, &y, y, b);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\ty=a; y+b, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  y = b;
  fastsum_f64(&x, &y, a, y);
  nfails += 1 - (int)(x==1001.0 && y==DBL_EPSILON);
  if (verbose)
    printf("\ty=b; a+y, x==1001 && y==eps  : '%c'\n", OK(x==1001.0 && y==DBL_EPSILON));

  return nfails;
}

int test_fastsum_extf64(int verbose)
{
  int nfails = 0;
  double x, y;
  double ah = 1e3;
  double al = -DBL_EPSILON;
  double bh = 1.0 + DBL_EPSILON;
  double bl = 0.0;

  if (verbose)
    printf("fastsum_extf64: (1000.0, -eps) + (1.0+eps, 0.0)\n");

  fastsum_f64_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x==1001.0 && y==0.0);
  if (verbose)
    printf("\t|a|>|b|, x==1001 && y==0.0  : '%c'\n", OK(x==1001.0 && y==0.0));
  return nfails;
}

int test_twoprod_f64(int verbose)
{
  int nfails = 0;
  double x, y;
  double a = 1.0 - DBL_EPSILON;
  double b = 1.0 + DBL_EPSILON;
  const double y_exp = -DBL_EPSILON*DBL_EPSILON;

  if (verbose)
    printf("twoprod_f64: (1.0 - eps) * (1.0 + eps)\n");

  twoprod_f64(&x, &y, a, b);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\t     a*b; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  twoprod_f64(&x, &y, b, a);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\t     b*a; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  x = a;
  twoprod_f64(&x, &y, x, b);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\tx=a; x*b; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  x = b;
  twoprod_f64(&x, &y, a, x);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\tx=b; a*x; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  y = a;
  twoprod_f64(&x, &y, y, b);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\ty=a; y*b; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  y = b;
  twoprod_f64(&x, &y, a, y);
  nfails = 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\ty=b; a*x; x==1.0 && y==-eps*eps  : '%c'\n", OK(x==1.0 && y==y_exp));

  return nfails;
}

int test_twoprod_extf64(int verbose)
{
  int nfails = 0;
  double x, y;
  double ah = 1.0 - DBL_EPSILON;
  double al = DBL_EPSILON*DBL_EPSILON/2.0;
  double bh = 1.0 + DBL_EPSILON;
  double bl = DBL_EPSILON*DBL_EPSILON/2.0;
  const double y_exp = 0.0;

  if (verbose)
    printf("twoprod_extf64: (1.0 - eps, eps*eps/2.0) * (1.0 + eps, eps*eps/2.0)\n");

  twoprod_f64_ext(&x, &y, ah, al, bh, bl);
  nfails += 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\ta*b; x==1.0 && y==0.0       : '%c'\n", OK(x==1.0 && y==y_exp));

  twoprod_f64_ext(&x, &y, bh, bl, ah, al);
  nfails += 1 - (int)(x==1.0 && y==y_exp);
  if (verbose)
    printf("\tb*a; x==1.0 && y==0.0       : '%c'\n", OK(x==1.0 && y==y_exp));
  return nfails;
}

int test_twodiv_f64(int verbose)
{
  int y_ok, x_ok, nfails = 0;
  double x, y;
  double b = 33.0;
  double a = 1.0 - DBL_EPSILON;

  if (verbose)
    printf("twodiv_f64: (1.0 - eps)/33.0\n");

  // approximate error term is |a/b - x| <= eps*|a|
  //     ==> |y| <= eps*|a| && |a-b*x| <= eps*|a|
  twodiv_f64(&x, &y, a, b);
  x_ok = fabs(a-b*x) <= DBL_EPSILON*fabs(a);
  y_ok = fabs(y)     <= DBL_EPSILON*fabs(a);

  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t     |a-b*x| <= eps*|a| && |y| <= eps*|a|  : '%c'\n", OK(x_ok && y_ok));

  x = a;
  twodiv_f64(&x, &y, x, b);
  x_ok = fabs(a-b*x) <= DBL_EPSILON*fabs(a);
  y_ok = fabs(y)     <= DBL_EPSILON*fabs(a);

  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\tx=a; |a-b*x| <= eps*|a| && |y| <= eps*|a|  : '%c'\n", OK(x_ok && y_ok));

  y = a;
  twodiv_f64(&x, &y, y, b);
  x_ok = fabs(a-b*x) <= DBL_EPSILON*fabs(a);
  y_ok = fabs(y)     <= DBL_EPSILON*fabs(a);

  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\ty=a; |a-b*x| <= eps*|a| && |y| <= eps*|a|  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}

int test_split_f64(int verbose)
{
  int ok, nfails = 0;
  double x, y;
  double a = 2.0*__factor_f64 + __factor_f64/2.0;

  if (verbose)
    printf("split_f64: 2*fact32 + fact32/2\n");

  split_f64(&x, &y, a);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\t     x+y = a  : '%c'\n", OK(ok));

  x = a;
  split_f64(&x, &y, x);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\tx=a; x+y = x  : '%c'\n", OK(ok));

  y = a;
  split_f64(&x, &y, y);
  ok = x+y == a;
  nfails += 1 - ok;
  if (verbose)
    printf("\ty=a; x+y = y  : '%c'\n", OK(ok));
  return nfails;
}

#if __HAVE_SIMD32X4
int test_twosum_f32x4(int verbose)
{
  int x_ok, y_ok, nfails = 0;
  float32x4_t x, y;
  float32x4_t a = set1_f32x4(1e3);
  float32x4_t b = set1_f32x4(1.0 + FLT_EPSILON);
  float32x4_t x_exp = set1_f32x4(1001.0);
  float32x4_t y_exp = set1_f32x4(FLT_EPSILON);

  if (verbose)
    printf("twosum_f32x4: |a| > |b|, 1000.0 + (1.0 + eps)\n");

  twosum_f32x4(&x, &y, a, b);
  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t     a+b, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));

  twosum_f32x4(&x, &y, b, a);
  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);

  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t     b+x, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));

  x = a;
  twosum_f32x4(&x, &y, x, b);
  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\tx=a; x+b, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));

  x = b;
  twosum_f32x4(&x, &y, x, b);
  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\tx=b; a+x, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));


  return nfails;
}

int test_twosum_extf32x4(int verbose)
{
  int x_ok, y_ok, nfails = 0;
  float32x4_t x, y;
  float32x4_t ah = set1_f32x4(1e3);
  float32x4_t al = set1_f32x4(-FLT_EPSILON);
  float32x4_t bh = set1_f32x4(1.0 + FLT_EPSILON);
  float32x4_t bl = set1_f32x4(0.0);
  float32x4_t y_exp = set1_f32x4(0.0);
  float32x4_t x_exp = set1_f32x4(1001.0);

  if (verbose)
    printf("twosum_extf32x4: (1000.0, -eps) + (1.0+eps, 0.0)\n");

  twosum_f32x4_ext(&x, &y, ah, al, bh, bl);

  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|>|b|, x==1001 && y==0.0  : '%c'\n", OK(x_ok && y_ok));

  twosum_f32x4_ext(&x, &y, bh, bl, ah, al);
  x_ok = eq_f32x4(x, x_exp);
  y_ok = eq_f32x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|<|b|, x==1001 && y==0.0  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}
#endif



#ifdef __HAVE_SIMD64X2
int test_twosum_f64x2(int verbose)
{
  int x_ok, y_ok, nfails = 0;
  float64x2_t x, y;
  float64x2_t a = set1_f64x2(1e3);
  float64x2_t b = set1_f64x2(1.0 + DBL_EPSILON);

  if (verbose)
    printf("twosum_f64x2: 1000.0 + (1.0 + eps)\n");

  twosum_f64x2(&x, &y, a, b);

  x_ok = x[0]==1001.0 && x[1]==1001.0;
  y_ok = y[0]==DBL_EPSILON && y[1]==DBL_EPSILON;
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|>|b|, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));

  twosum_f64x2(&x, &y, b, a);

  x_ok = x[0]==1001.0 && x[1]==1001.0;
  y_ok = y[0]==DBL_EPSILON && y[1]==DBL_EPSILON;
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|<|b|, x==1001 && y==eps  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}
#endif


#ifdef __HAVE_SIMD32X8
int test_twosum_f32x8(int verbose)
{
  int x_ok, y_ok, nfails = 0;
  float32x8_t x, y;
  float32x8_t a = set1_f32x8(1e3);
  float32x8_t b = set1_f32x8(1.0 + FLT_EPSILON);
  float32x8_t x_exp = set1_f32x8(1001.0);
  float32x8_t y_exp = set1_f32x8(FLT_EPSILON);

  if (verbose)
    printf("twosum_f32x8: 1000.0 + (1.0 + eps)\n");

  twosum_f32x8(&x, &y, a, b);
  x_ok = eq_f32x8(x, x_exp);
  y_ok = eq_f32x8(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|>|b|, X==1001 && Y==eps  : '%c'\n", OK(x_ok && y_ok));


  twosum_f32x8(&x, &y, b, a);
  x_ok = eq_f32x8(x, x_exp);
  y_ok = eq_f32x8(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|<|b|, X==1001 && Y==eps  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}
#endif

#ifdef __HAVE_SIMD64X4
int test_twosum_f64x4(int verbose)
{
  int x_ok, y_ok, nfails = 0;
  float64x4_t x, y;
  float64x4_t a = set1_f64x4(1e3);
  float64x4_t b = set1_f64x4(1.0 + DBL_EPSILON);
  float64x4_t x_exp = set1_f64x4(1001.0);
  float64x4_t y_exp = set1_f64x4(DBL_EPSILON);

  if (verbose)
    printf("twosum_f64x4: 1000.0 + (1.0 + eps)\n");

  twosum_f64x4(&x, &y, a, b);
  x_ok = eq_f64x4(x, x_exp);
  y_ok = eq_f64x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|>|b|, X==1001 && Y==eps  : '%c'\n", OK(x_ok && y_ok));

  twosum_f64x4(&x, &y, b, a);
  x_ok = eq_f64x4(x, x_exp);
  y_ok = eq_f64x4(y, y_exp);
  nfails += 1 - (int)(x_ok && y_ok);
  if (verbose)
    printf("\t|a|<|b|, X==1001 && Y==eps  : '%c'\n", OK(x_ok && y_ok));
  return nfails;
}

#if 0
// A = {a0, a1, a2, a3}
void hsum_f64x4_ext(double *s, double *r, float64x4_t A, float64x4_t B)
{
  double q;
  float64_t S0, S1, R0, R1;
  q = hsum_f64x4(B);
  // S0 = {a2, a3, a0, a1}
  S0 = hflip_f64x4(A);

  // S1 = {s0, s1, s2, s3} = {a2+a0, a3+a1, a2+a0, a3+a1}
  twosum_f64x4(&S1, &R1, A, S0);

  // S0 = {s1, s0, s3, s2} = {a3+a1, a2+a0, a3+a1, a2+a0}
  S0 = pflip_f64x4(S1);
  R0 = pflip_f64x4(R1);
  q += R0[0] + R1[0];
  twosum_f64x4(S1, R1, S1, S0);
}
#endif

#endif


#define TEST(nerr, func)					\
  do {								\
    int n = func(verbose);					\
    if (!verbose)						\
      printf("%4s\t%s\n", n == 0 ? "OK" : "FAIL", #func);	\
    nerr += n;							\
  } while (0)

int main(int argc, char **argv)
{
  int nfails = 0, verbose = 0;

  if (argc > 1)
    verbose = 1;

  TEST(nfails, test_twosum_f32);
  TEST(nfails, test_twosum_extf32);
  TEST(nfails, test_fastsum_f32);
  TEST(nfails, test_fastsum_extf32);
  TEST(nfails, test_twoprod_f32);
  TEST(nfails, test_twoprod_extf32);
  TEST(nfails, test_twodiv_f32);
  TEST(nfails, test_split_f32);

  TEST(nfails, test_twosum_f64);
  TEST(nfails, test_twosum_extf64);
  TEST(nfails, test_fastsum_f64);
  TEST(nfails, test_fastsum_extf64);
  TEST(nfails, test_twoprod_f64);
  TEST(nfails, test_twoprod_extf64);
  TEST(nfails, test_twodiv_f64);

#ifdef __HAVE_SIMD32X4
  TEST(nfails, test_twosum_f32x4);
#endif

#ifdef __HAVE_SIMD64X2
  TEST(nfails, test_twosum_f64x2);
#endif

#ifdef __HAVE_SIMD32X8
  TEST(nfails, test_twosum_f32x8);
#endif

#ifdef __HAVE_SIMD64X4
  TEST(nfails, test_twosum_f64x4);
#endif
  return nfails;
}
