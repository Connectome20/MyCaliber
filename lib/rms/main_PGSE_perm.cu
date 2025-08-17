//
//  main.cpp
//  diffusion_cylinder_exchange
//
//  Update Journal:
//  -- 06/26/2017: massive job version
//  -- 07/20/2017: do not divide DW signal & cumulants by b0 signal, record particle number in each compartment
//  -- 04/27/2019: diffusion in spheres with permeable membrane
//  -- 05/01/2019: implement cuda
//  -- 06/18/2019: re-write sphere code to cylinder code for cuda
//  -- 02/20/2020: re-write coaxial cylinder code to single-layer cylinder code for cuda
//  -- 03/06/2020: fix the bug for permeation step along z-axis
//  -- 12/14/2023: fix the mechanism of membrane permeation
//  -- 11/02/2024: re-write to coaxial cylinders without permeation
//  -- 07/28/2025: include permeation between myelin water layers
//
//  Created by Hong-Hsi Lee in February, 2017.
//


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <complex>
#include <string>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;
    
#define Pi 3.14159265
#define timepoints 1000
#define NDelta_max 500
#define nbin 200
#define nite 100

// ********** diffusion library **********

__device__ double pow2 (const double &x) {
    return (x*x);
}

__device__ void pixPosition ( const double x_in[], const unsigned int &NPix, int xPix[] ) {
    double x[2]={0}; x[0]=x_in[0]; x[1]=x_in[1]; //x[2]=x_in[2];
    
    if ( x[0]<0 ) { x[0]+=1; }
    if ( x[0]>1 ) { x[0]-=1; }
    
    if ( x[1]<0 ) { x[1]+=1; }
    if ( x[1]>1 ) { x[1]-=1; }
    
//    if ( x[2]<0 ) { x[2]+=1; }
//    if ( x[2]>1 ) { x[2]-=1; }
    
    xPix[0]=floor(x[0]*NPix);
    xPix[1]=floor(x[1]*NPix);
//    xPix[2]=floor(x[2]*NPix);
}

__device__ void translateXc ( const double x[], double xc[] ) {
    // Translate circle center xc to make it as close to the position x as possible
    double ti=0, tj=0;
    double d2 = pow2(x[0]-xc[0])+pow2(x[1]-xc[1]), d2Tmp=0;
    int ii[2]={0}, jj[2]={0};
    ii[1]=2*(xc[0]<0.5)-1;
    jj[1]=2*(xc[1]<0.5)-1;
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            if ( i==0 & j==0 ){ continue; }
            d2Tmp=pow2(x[0]-xc[0]-ii[i])+pow2(x[1]-xc[1]-jj[j]);
            if (d2Tmp<d2) {
                d2=d2Tmp;
                ti=ii[i];
                tj=jj[j];
            }
        }
    }
    xc[0]+=ti;
    xc[1]+=tj;
}

__device__ int whichlayer( const double x[], const double &rc, const double &lm) {
    return ( floor( (sqrt( pow2(x[0]-0.5)+pow2(x[1]-0.5) )-rc)/lm ) );
}

__device__ bool inCyl ( const double x[], const double xc_in[], const double &rc, const bool &translateFlag ) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // If the point x is in the circle (xc,rc), return 1; if not, return 0.
    
    // Translate circle center xc to make it as close to the position xt as possible
    if ( translateFlag ) { translateXc(x,xc); }
    
    return ( ( pow2(x[0]-xc[0])+pow2(x[1]-xc[1]) ) <= rc*rc );
}

__device__ bool stepE2A (const double xi[], const double xt[], const double xc_in[], const double &rc, double &t, const bool &translateFlag) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // If segment(xi,xt) overlaps circle (xc,rc), return 1; if not, return 0.
    
    // Translate circle center xc to make it as close to the position xt as possible
    if ( translateFlag ) { translateXc(xt,xc); }
    
    t=-( (xi[0]-xc[0])*(xt[0]-xi[0])+(xi[1]-xc[1])*(xt[1]-xi[1]) ) / ( pow2(xi[0]-xt[0]) + pow2(xi[1]-xt[1]) );
    
    // If xt is in the cell, segment overlaps the circle.
    if ( ( pow2(xt[0]-xc[0])+pow2(xt[1]-xc[1]) ) <= rc*rc ) {
        return true;
    } else {
        // L: a line connecting xi and xt
        // xl: a point on L closest to xc, xl = xi + (xt-xi)*t
        // d: distance of xc to L (or xl)
        // Reference: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        double xl[2]={0};
        xl[0]=xi[0]+(xt[0]-xi[0])*t;
        xl[1]=xi[1]+(xt[1]-xi[1])*t;
        double d2=pow2(xl[0]-xc[0])+pow2(xl[1]-xc[1]);
        
        // If d>rc, segment does not overlap the circle.
        if (d2>rc*rc) {
            return false;
        } else {
            // xl is in ICS, but xi and xt are both in ECS.
            return ( ( (xi[0]-xl[0])*(xt[0]-xl[0])+(xi[1]-xl[1])*(xt[1]-xl[1]) ) <= 0 );
        }
    }
}

__device__ void elasticECS (const double x[], const double v[], const double &dx, const double &dz, const double xc_in[], const double &rc, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Elastic collision from x in ECS onto a cell membrane (xc,rc) with a direction v and a step size dx.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dx*v[0];
    xTmp[1]=x[1]+dx*v[1];
    xTmp[2]=x[2]+dz*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                    // Does not encounter the cell membrane
        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=min(t1,t2);
        if ( (t>=dx) | (t<0) ) {        // Does encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xm-xc), outward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xm[0]-xc[0])/t1;
            n[1]=(xm[1]-xc[1])/t1;
            
            // v' = v - 2*dot(v,n)*n
            t1=v[0]*n[0]+v[1]*n[1];
            n[0]=v[0]-2*t1*n[0];
            n[1]=v[1]-2*t1*n[1];
            
            // xt = xm + (dx-t)*v'
            xt[0]=xm[0]+(dx-t)*n[0];
            xt[1]=xm[1]+(dx-t)*n[1];
            xt[2]=xTmp[2];
        }
    }
}

__device__ void permeateE2I (const double x[], const double v[], const double &dxEX, const double &dzEX, const double xc_in[], const double &rc, const double &dxIN, const double &dzIN, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Permeation from x in ECS into a cell (xc,rc) with a direction v and a step size dxEX.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dxEX*v[0];
    xTmp[1]=x[1]+dxEX*v[1];
    xTmp[2]=x[2]+dzEX*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                    // Does not encounter the cell membrane
        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=min(t1,t2);
        if ( (t>=dxEX) | (t<0) ) {      // Does encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // Diffusion across the cell membrane needs the adjustment of the step size.
            // xt = xm + (1-t/dxEX)*dxIN * v
            xt[0]=xm[0]+(1-t/dxEX)*dxIN*v[0];
            xt[1]=xm[1]+(1-t/dxEX)*dxIN*v[1];
            // xt = x + t/dxEX*dzEX*vz + (1-t/dxEX)*dzIN*vz
            xt[2]=x[2] + t/dxEX*dzEX*v[2] + (1-t/dxEX)*dzIN*v[2];
        }
    }
}

__device__ void elasticICS (const double x[], const double v[], const double &dx, const double &dz, const double xc_in[], const double &rc, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Elastic collision from x in ICS onto a cell membrane (xc,rc) with a direction v and a step size dx.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dx*v[0];
    xTmp[1]=x[1]+dx*v[1];
    xTmp[2]=x[2]+dz*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;

    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                     // Walker is right on the surface and diffuses tangent to the surface
//        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        xt[0]=x[0]; xt[1]=x[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=max(t1,t2);
        if ( t>=dx ) {                  // Does not encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        }
        else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xm-xc), outward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xm[0]-xc[0])/t1;
            n[1]=(xm[1]-xc[1])/t1;
            
            // v' = v - 2*dot(v,n)*n
            t1=v[0]*n[0]+v[1]*n[1];
            n[0]=v[0]-2*t1*n[0];
            n[1]=v[1]-2*t1*n[1];
            
            // xt = xm + (dx-t)*v'
            xt[0]=xm[0]+(dx-t)*n[0];
            xt[1]=xm[1]+(dx-t)*n[1];
            xt[2]=xTmp[2];
        }
    }
}

__device__ void permeateI2E (const double x[], const double v[], const double &dxIN, const double &dzIN, const double xc_in[], const double &rc, const double &dxEX, const double &dzEX, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Permeation from x in ICS out of the cell (xc,rc) with a direction v and a step size dxIN.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dxIN*v[0];
    xTmp[1]=x[1]+dxIN*v[1];
    xTmp[2]=x[2]+dzIN*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                     // Walker is right on the surface and diffuses tangent to the surface
        xt[0]=x[0]; xt[1]=x[1]; xt[2]=x[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=max(t1,t2);
        if ( t>=dxIN ) {                 // Does not encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                         // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // Diffusion across the cell membrane needs the adjustment of the step size.
            // xt = xm + (1-t/dxIN)*dxEX * v
            xt[0]=xm[0]+(1-t/dxIN)*dxEX*v[0];
            xt[1]=xm[1]+(1-t/dxIN)*dxEX*v[1];
            // xt = x + t/dxIN*dzIN*vz + (1-t/dxIN)*dzEX*vz
            xt[2]=x[2] + t/dxIN*dzIN*v[2] + (1-t/dxIN)*dzEX*v[2];
        }
    }
}

// ********** cuda kernel **********
__device__ double atomAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
        
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void propagate(
    curandStatePhilox4_32_10_t *state, 
    double *dx2, 
    double *dx4, 
    double *NParICS, 
    double *NParBin,
    double *sig, 
    const int TN, 
    const int NPar, 
    const double res, 
    const double step, 
    const double stepz,
    const double prob,
    const unsigned int Nbtab,
    const unsigned int NDelta,
    const double rCir, 
    const double lm,
    const unsigned int Nm,
    const double *btab,
    const double *DELdel,
    const double dt){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t localstate=state[idx];
    
    int Tstep=TN/timepoints;
    
    for (int k=idx; k<NPar; k+=stride){
        double rCirTmp = 0;
        double xPar[3]={0}, xCirTmp[3]={0};
        int nlayer_i = 0;                                   // current layer label
        int nlayer_j = 0;                                   // temporary layer lable
                
        double xi[3]={0}, xt[3]={0}, xTmp[3]={0};           // Particle position
        double xCollision[3]={0};                           // Position after collision
        double vrand=0;                                     // Random number
        int tidx=0, bidx=0;
        
        double vp[3]={0};                                   // Nomalized diffusion velocity
        bool iterateFlag=false;                             // true: choose another direction and leap again, false: finish the iteration
        int ite=0;                                          // number of iterations
        // bool inLayerFlag=false;                             // true: in the same layer, false: not in the same layer
        
        double dx=0, dy=0, dz=0;

        double phase = 0; // wide pulse
        double x1[NDelta_max*3] = {0};
        double ttmp = 0;

                
        //********** Initialize Walker Positions *********
        while (1){
            xPar[0]=curand_uniform_double(&localstate);
            xPar[1]=curand_uniform_double(&localstate);
            xPar[2]=curand_uniform_double(&localstate);

            // Identify the layer label, 0 to Nm-1
            nlayer_j = whichlayer(xPar, rCir, lm);
            if ( nlayer_j>=0 && nlayer_j < Nm) {
                xi[0]=xPar[0]; xi[1]=xPar[1]; xi[2]=xPar[2];
                nlayer_i = nlayer_j;
                break;
            }
        }
        
        // ********** Simulate diffusion **********
        xt[0]=xi[0]; xt[1]=xi[1]; xt[2]=xi[2];
        for (int i=0; i<TN; i++){
            nlayer_j = whichlayer(xt, rCir, lm);
            // if ( nlayer==nlayer_i ) { 
            //     inLayerFlag=true; 
            // } else {
            //     inLayerFlag=false;
            // }
            
            iterateFlag=true; ite=0;
            // ********** One step **********
            while (iterateFlag & (ite<nite)) {
                // if ( !inLayerFlag ) {
                //     // Case 1, not in the same layer: back to initial
                //     xt[0]=xi[0]; xt[1]=xi[1]; xt[2]=xi[2];
                // } else {

                // Primitive position after diffusion
                vrand=curand_uniform_double(&localstate);
                vp[0]=cos(2*Pi*vrand);
                vp[1]=sin(2*Pi*vrand);
                vrand=curand_uniform_double(&localstate);
                vp[2]=2.0*(vrand<0.5)-1;
                xTmp[0]=xt[0]+step *vp[0];
                xTmp[1]=xt[1]+step *vp[1];
                xTmp[2]=xt[2]+stepz*vp[2];
                
                nlayer_j = whichlayer(xTmp, rCir, lm);
                
                if ( nlayer_j==nlayer_i ) {
                    // Case 1 Walker does not encounter the cell membrane
                    xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                    iterateFlag=false; // inLayerFlag=true;
                } else if (nlayer_j < nlayer_i) {
                    // Case 2 Walker encounters the inner cell membrane
                    // Similar to diffusion in ECS
                    xCirTmp[0] = 0.5;
                    xCirTmp[1] = 0.5;
                    rCirTmp = rCir + nlayer_i*lm;

                    vrand=curand_uniform_double(&localstate);
                    if (vrand<prob && nlayer_j>=0) {
                        // Case 2.1 Permeation from outer to inner layers
                        // Similar to permeation from ECS to ICS
                        permeateE2I(xt, vp, step, stepz, xCirTmp, rCirTmp, step, stepz, 0, xTmp);
                        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                        nlayer_i = nlayer_j;
                        iterateFlag=false;
                    }
                    else {
                        // Case 2.2 Elastic collision in inner layer
                        // Similar to elastic collision in ECS
                        elasticECS(xt, vp, step, stepz, xCirTmp, rCirTmp, 0, xCollision);
                        
                        // Use xTmp to save the present position
                        xTmp[0]=xt[0]; xTmp[1]=xt[1]; xTmp[2]=xt[2];
                        
                        // Case 2.2.1 Renew the step for the elastic collision
                        xt[0]=xCollision[0]; xt[1]=xCollision[1]; xt[2]=xCollision[2];
                        iterateFlag=false;
                        
                        // Case 2.2.2 Cancel this step and choose another direction if bouncing twice
                        nlayer_j = whichlayer(xt, rCir, lm);
                        if ( nlayer_j!=nlayer_i ) {
                            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                            iterateFlag=true; ite++;
                        }
                        
                        // if ( iterateFlag==false ) {
                        //     inLayerFlag=true;
                        // }
                    }

                } else if (nlayer_j > nlayer_i) {
                    // Case 3 Walker encounters the outer cell membrane
                    // Similar to diffusion in ICS
                    xCirTmp[0] = 0.5;
                    xCirTmp[1] = 0.5;
                    rCirTmp = rCir + (nlayer_i+1)*lm;

                    vrand=curand_uniform_double(&localstate);
                    if (vrand<prob && nlayer_j<Nm) {
                        // Case 3.1 Permeation from inner to outer layers
                        // Similar to permeation from ICS to ECS
                        permeateI2E(xt, vp, step, stepz, xCirTmp, rCirTmp, step, stepz, 0, xTmp);
                        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                        nlayer_i = nlayer_j;
                        iterateFlag=false;
                    }
                    else {
                        // Case 3.2 Elastic collision in outer layer
                        // Similar to elastic collision in ICS
                        elasticICS(xt, vp, step, stepz, xCirTmp, rCirTmp, 0, xCollision);
                           
                        // Use xTmp to save the present position
                        xTmp[0]=xt[0]; xTmp[1]=xt[1]; xTmp[2]=xt[2];
                        
                        // Case 3.2.1 Renew the step for the elastic collision
                        xt[0]=xCollision[0]; xt[1]=xCollision[1]; xt[2]=xCollision[2];
                        iterateFlag=false;
                        
                        // Case 3.2.2 Cancel this step and choose another direction if bouncing twice
                        nlayer_j = whichlayer(xt, rCir, lm);
                        if ( nlayer_j!=nlayer_i ) {
                            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                            iterateFlag=true; ite++;
                        }
                        
                        // if ( iterateFlag==false ) {
                        //     inLayerFlag=true;
                        // }
                    }
                    
                }
                // }
            }

            if (ite==nite) {
                printf("Run out of iterations.\n");
            }
            
            // ********** End one step **********
            
            dx=(xt[0]-xi[0])*res;
            dy=(xt[1]-xi[1])*res;
            dz=(xt[2]-xi[2])*res;

            ttmp = static_cast<double>(i+1)*dt;

            // add phase 
            for(int j = 0; j < NDelta; j++){
                // Wide pulse
                if(ttmp <= DELdel[j*2+1]){
                    x1[3*j]+=dx; x1[3*j+1]+=dy; x1[3*j+2]+=dz;
                } // 1st
                else if ( (ttmp > DELdel[j*2]) & (ttmp <= (DELdel[j*2]+DELdel[j*2+1])) ){
                    x1[3*j]-=dx; x1[3*j+1]-=dy; x1[3*j+2]-=dz;
                } // 2nd
            } // j
            
            if ( (i%Tstep) ==0 ) { // Save moment tensor for dx^2 and dx^4, and signal for the b-table
                tidx=i/Tstep;

                if ( nlayer_i>=0 && nlayer_i<Nm ) { atomAdd(&NParICS[tidx],1); }
                bidx=floor( sqrt( pow2(xt[0]-0.5) + pow2(xt[1]-0.5) )*static_cast<double>(nbin*2) );
                if (bidx<nbin) {
                    atomAdd(&NParBin[nbin*tidx+bidx],1);
                } else {
                    atomAdd(&NParBin[nbin*tidx+nbin-1],1);
                }
                
                atomAdd(&dx2[6*tidx+0],dx*dx);
                atomAdd(&dx2[6*tidx+1],dx*dy);
                atomAdd(&dx2[6*tidx+2],dx*dz);
                atomAdd(&dx2[6*tidx+3],dy*dy);
                atomAdd(&dx2[6*tidx+4],dy*dz);
                atomAdd(&dx2[6*tidx+5],dz*dz);
                
                atomAdd(&dx4[15*tidx+0],dx*dx*dx*dx);
                atomAdd(&dx4[15*tidx+1],dx*dx*dx*dy);
                atomAdd(&dx4[15*tidx+2],dx*dx*dx*dz);
                atomAdd(&dx4[15*tidx+3],dx*dx*dy*dy);
                atomAdd(&dx4[15*tidx+4],dx*dx*dy*dz);
                atomAdd(&dx4[15*tidx+5],dx*dx*dz*dz);
                atomAdd(&dx4[15*tidx+6],dx*dy*dy*dy);
                atomAdd(&dx4[15*tidx+7],dx*dy*dy*dz);
                atomAdd(&dx4[15*tidx+8],dx*dy*dz*dz);
                atomAdd(&dx4[15*tidx+9],dx*dz*dz*dz);
                atomAdd(&dx4[15*tidx+10],dy*dy*dy*dy);
                atomAdd(&dx4[15*tidx+11],dy*dy*dy*dz);
                atomAdd(&dx4[15*tidx+12],dy*dy*dz*dz);
                atomAdd(&dx4[15*tidx+13],dy*dz*dz*dz);
                atomAdd(&dx4[15*tidx+14],dz*dz*dz*dz);
                
            }

            if ( i==(TN-1) ) {
                for (int j=0; j<NDelta; j++){
                    for (int jj=0; jj<Nbtab; jj++){
                        phase = sqrt( btab[jj*4]/(DELdel[j*2]-DELdel[j*2+1]/3.0) )/DELdel[j*2+1] * (x1[j*3]*btab[jj*4+1] + x1[j*3+1]*btab[jj*4+2] + x1[j*3+2]*btab[jj*4+3])*dt;
                        atomAdd(&sig[j*Nbtab+jj],cos(phase));
                    }
                }
            }
            
        }
    }
    state[idx]=localstate;
}

    
//********** Define tissue parameters **********

int main(int argc, char *argv[]) {
    
    clock_t begin=clock();
    clock_t end=clock();
    
    // Define index number
    int i=0, j=0;
    
    //********** Load mictostructure **********
    
    double dt=0;                // Time step in ms
    int TN=0;                   // Number of time steps
    int NPar=0;                 // Number of time points to record
    double Din=0;               // Diffusion coefficient inside the axon in µm^2/ms
    double kappa=0;             // Permeability, um/ms
    int thread_per_block=0;     // Number of threads per block
    double rCir=0;              // Inner radius/FOV, no unit
    double lm=0;                // Distance bewteen layer/FOV, no unit
    int Nm=0;                   // Total number of layers
    double res=0;               // FOV in µm
    
    // simulation parameter
    ifstream myfile0 ("simParamInput.txt", ios::in);
    myfile0>>dt; myfile0>>TN; myfile0>>NPar;
    myfile0>>Din;
    myfile0>>kappa;
    myfile0>>thread_per_block;
    myfile0>>rCir;
    myfile0>>lm;
    myfile0>>Nm;
    myfile0>>res;
    myfile0.close();
    
    double step=sqrt(4.0*dt*Din);     // Step size in ICS in µm
    double stepz=sqrt(2.0*dt*Din);    // Step size along z-dirction in µm
    
    // Number of PGSE b-table
    unsigned int Nbtab=0;
    ifstream myfile9 ("gradient_Nbtab.txt", ios::in);
    myfile9>>Nbtab;
    myfile9.close();

    // Gradient parameters of PGSE: b, gx, gy, gz
    thrust::host_vector<double> btab(Nbtab*4);
    ifstream myfile10 ("gradient_btab.txt", ios::in);
    for (i=0; i<Nbtab*4; i++){
        myfile10>>btab[i];
    }
    myfile10.close();

    // Number of PGSE (Delta, delta)
    unsigned int NDelta=0;
    ifstream myfile11 ("gradient_NDelta.txt", ios::in);
    myfile11>>NDelta;
    myfile11.close();
    
    // Gradient parameters of PGSE: Delta, delta
    thrust::host_vector<double> DELdel(NDelta*2);
    ifstream myfile12 ("gradient_DELdel.txt", ios::in);
    for (i=0; i<NDelta*2; i++){
        myfile12>>DELdel[i];
        // cout<<DELdel[i]<<endl;
    }
    myfile12.close();


    
    // Diffusion time, cumulant
    thrust::host_vector<double> TD(timepoints);
    for (i=0; i<timepoints; i++){
        TD[i]=(i*(TN/timepoints)+1)*dt;
    }
    
    //********** Initialize Particle Positions in IAS *********
    const double prob = (Pi/4.0*step*kappa/Din)/( 1 + Pi/4.0*step*kappa/Din);
    step/=res;                            // Normalize the step size with the voxel size
    stepz/=res;
    
    // ********** Simulate diffusion **********
    
    // Initialize seed
    unsigned long seed=0;
    FILE *urandom;
    urandom = fopen("/dev/random", "r");
    fread(&seed, sizeof (seed), 1, urandom);
    fclose(urandom);
    
    // Initialize state of RNG
    int blockSize = thread_per_block;
    int numBlocks = (NPar + blockSize - 1) / blockSize;
    cout<<numBlocks<<endl<<blockSize<<endl;
    
    thrust::device_vector<curandStatePhilox4_32_10_t> devState(numBlocks*blockSize);
    setup_kernel<<<numBlocks, blockSize>>>(devState.data().get(),seed);
    
    // Initialize output
    thrust::host_vector<double> dx2(timepoints*6);
    thrust::host_vector<double> dx4(timepoints*15);
    thrust::host_vector<double> sig(NDelta*Nbtab);
    for (i=0;i<timepoints*6;i++){ dx2[i]=0; }
    for (i=0;i<timepoints*15;i++){ dx4[i]=0; }
    for (i=0;i<NDelta*Nbtab;i++) { sig[i]=0; }

    thrust::host_vector<double> NParICS(timepoints);
    thrust::host_vector<double> NParBin(timepoints*nbin);
    for (i=0;i<timepoints;i++){ NParICS[i]=0; }
    for (i=0;i<timepoints*nbin;i++){ NParBin[i]=0; }
    
    // Move data from host to device
    thrust::device_vector<double> d_dx2=dx2;
    thrust::device_vector<double> d_dx4=dx4;
    thrust::device_vector<double> d_sig=sig;
    thrust::device_vector<double> d_btab=btab;
    thrust::device_vector<double> d_DELdel=DELdel;

    thrust::device_vector<double> d_NParICS=NParICS;
    thrust::device_vector<double> d_NParBin=NParBin;
    
    // Parallel computation
    begin=clock();
    propagate<<<numBlocks, blockSize>>>(devState.data().get(), 
                                        d_dx2.data().get(), 
                                        d_dx4.data().get(),
                                        d_NParICS.data().get(), 
                                        d_NParBin.data().get(),
                                        d_sig.data().get(), 
                                        TN, NPar, res, step, stepz, prob,
                                        Nbtab, NDelta, 
                                        rCir, lm, Nm, 
                                        d_btab.data().get(), 
                                        d_DELdel.data().get(),
                                        dt);
    cudaDeviceSynchronize();
    end=clock();
    cout << "Done! Elpased time "<<double((end-begin)/CLOCKS_PER_SEC) << " s"<< endl;
    
    thrust::copy(d_dx2.begin(), d_dx2.end(), dx2.begin());
    thrust::copy(d_dx4.begin(), d_dx4.end(), dx4.begin());
    thrust::copy(d_sig.begin(), d_sig.end(), sig.begin());

    thrust::copy(d_NParICS.begin(), d_NParICS.end(), NParICS.begin());
    thrust::copy(d_NParBin.begin(), d_NParBin.end(), NParBin.begin());
    
    ofstream fdx2out("dx2_diffusion.txt");
    ofstream fdx4out("dx4_diffusion.txt");
    ofstream fsigout("sig_diffusion.txt");
    fdx2out.precision(15);
    fdx4out.precision(15);
    fsigout.precision(15);

    ofstream fNParICSout("NParICS.txt");
    ofstream fNParBinout("NParBin.txt");
    fNParICSout.precision(15);
    fNParBinout.precision(15);
    double dr = 0.5*res/nbin;

    for (i=0; i<timepoints; i++) {
        for (j=0; j<6; j++) {
            if (j==5) {
                fdx2out<<dx2[i*6+j]<<endl;
            } else {
                fdx2out<<dx2[i*6+j]<<"\t";
            }
        }
        for (j=0; j<15; j++) {
            if (j==14) {
                fdx4out<<dx4[i*15+j]<<endl;
            } else {
                fdx4out<<dx4[i*15+j]<<"\t";
            }
        }

        fNParICSout<<NParICS[i]<<endl;
        for (j=0; j<nbin; j++) {
            if (j==nbin-1){
                fNParBinout<<NParBin[i*nbin+j]/(Pi*dr*dr*(2*j+1))<<endl;
            } else {
                fNParBinout<<NParBin[i*nbin+j]/(Pi*dr*dr*(2*j+1))<<"\t";
            }
        }
        
    }
    for (i=0; i<NDelta*Nbtab; i++){
        fsigout<<sig[i]<<endl;
    }
    fdx2out.close();
    fdx4out.close();
    fsigout.close();

    fNParICSout.close();
    fNParBinout.close();
    
    ofstream paraout ("sim_para.txt");
    paraout<<dt<<endl<<TN<<endl<<NPar<<endl;
    paraout<<Din<<endl;
    paraout<<rCir<<endl<<lm<<endl<<Nm<<endl<<res<<endl;
    paraout<<kappa<<endl;
    paraout.close();
    
    ofstream TDout("diff_time.txt");
    for (i=0; i<timepoints; i++){
        TDout<<(i*(TN/timepoints)+1)*dt<<endl;
    }
    TDout.close();
}

