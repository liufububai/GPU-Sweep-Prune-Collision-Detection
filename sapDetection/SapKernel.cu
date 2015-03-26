#include <stdio.h>
#include <math.h>
#include "cutil_math.h"
#include "math_constants.h"
#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

#include "SapKernel.cuh"
#include "cudpp.h"

//! Absolute integer representation of a floating-point value
#define AIR(x)					(IR(x)&0x7fffffff)
//! Integer representation of a floating-point value.
#define IR(x)					((udword&)(x))
#define isPow2(x)			  (x&(x-1))==0
#define FLT_MAX         3.402823466e+38F 

texture<float4, 1, cudaReadModeElementType> cenTex;
texture<float4, 1, cudaReadModeElementType> velTex;
texture<float, 1, cudaReadModeElementType> posTex;
texture<int, 1, cudaReadModeElementType> hashTex;

__global__ void
axisVec(float *g_idata1, float *g_idata2, int* d_lamda)
{
     int axis, axis1;
     
     axis = g_idata1[0] >= g_idata2[0] ? 0 : 1;
     if (axis)
       axis = g_idata2[0] >= g_idata2[2] ? 1 : 2;
     else
       axis = g_idata1[0] >= g_idata2[2] ? 0 : 2; 
       
     if(axis == 0)
     { 
		 g_idata1[0] = 1.0;
		 g_idata1[1] = 0.0;
		 g_idata1[2] = 0.0;
		 axis1 = g_idata2[0] < g_idata2[2] ? 1 : 2;
	 }
	 else if(axis == 1)
	 { 
		 g_idata1[0] = 0.0;
		 g_idata1[1] = 1.0;
		 g_idata1[2] = 0.0;
		 axis1 = g_idata1[0] < g_idata2[2] ? 0 : 2; 
	 }
	 else
	 {
	     g_idata1[0] = 0.0;
		 g_idata1[1] = 0.0;
		 g_idata1[2] = 1.0;
		 axis1 = g_idata1[0] < g_idata2[0] ? 0 : 1;
	 }
	 d_lamda[0] = axis;
	 d_lamda[1] = axis1;
	 d_lamda[2] = 3 - axis - axis1;
}

__global__ void
axisVecMultiple(float *g_idata1, float *g_idata2, int* d_lamda)
{
     int axis, axis1;
     
     axis = g_idata1[0] >= g_idata2[0] ? 0 : 1;
     if (axis)
       axis = g_idata2[0] >= g_idata2[2] ? 1 : 2;
     else
       axis = g_idata1[0] >= g_idata2[2] ? 0 : 2; 
     //axis = 0;// z axis 2010. Oct. 26th.
     if(axis == 0)
     { 
		 g_idata1[0] = 1.0;
		 g_idata1[1] = 0.0;
		 g_idata1[2] = 0.0;
		 axis1 = 1;
	 }
	 else if(axis == 1)
	 { 
		 g_idata1[0] = 0.0;
		 g_idata1[1] = 1.0;
		 g_idata1[2] = 0.0;
		 axis1 = 0; 
	 }
	 else
	 {
	     g_idata1[0] = 0.0;
		 g_idata1[1] = 0.0;
		 g_idata1[2] = 1.0;
		 axis1 = 0;
	 }
	 d_lamda[0] = axis;
	 d_lamda[1] = axis1;
	 d_lamda[2] = 3 - axis - axis1;
}

__global__ void
projectSub(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id;
   __shared__ float mPoint;
   if(tx == 0)
   {
	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id = d_lamda[1];
	   //mPoint = d_matrix[id]/size; 
	   mPoint = 0.0; 
   }
    __syncthreads();
    
   if(id == 0){
	if(d_cen[th_id].x > mPoint)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
    }

   if(id == 1){
	if(d_cen[th_id].y > mPoint)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
    }
    
   if(id == 2){
	if(d_cen[th_id].z > mPoint)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
    }  

    d_Sorted[th_id] = th_id;
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
	{
		d_pos[th_id+1] =  FLT_MAX;
        d_Sorted[th_id+1] = th_id+1;
	}
}

__global__ void
projectSubMultiple(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2;
   if(tx == 0)
   {
	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0; 
	   mPoint2 = 0.0;
   }
    __syncthreads();
    
   if(id0 == 0){
	if(d_cen[th_id].y >= mPoint1 && d_cen[th_id].z >= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].y < mPoint1 && d_cen[th_id].z >= mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].y >= mPoint1 && d_cen[th_id].z < mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }

   if(id0 == 1){
	if(d_cen[th_id].x >= mPoint1 && d_cen[th_id].z >= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x < mPoint1 && d_cen[th_id].z >= mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x >= mPoint1 && d_cen[th_id].z < mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }
    
   if(id0 == 2){
	if(d_cen[th_id].x >= mPoint1 && d_cen[th_id].y >= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x < mPoint1 && d_cen[th_id].y >= mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x >= mPoint1 && d_cen[th_id].y < mPoint2)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }  
    
    d_Sorted[th_id] = th_id;
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
	{
		d_pos[th_id+1] =  FLT_MAX;
        d_Sorted[th_id+1] = th_id+1;
	}
}

__global__ void
projectSubMultiple9(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2, space;
   if(tx == 0)
   {
	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0; 
	   mPoint2 = 0.0;
	   space = interval/6;
   }
    __syncthreads();
    
   if(id0 == 0){
	if(d_cen[th_id].y <= mPoint1-space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].y <= mPoint1-space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z <= mPoint2-space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }

   if(id0 == 1){
	if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z <= mPoint2-space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;  
    }
    
   if(id0 == 2){
	if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].y > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y <= mPoint2-space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;  
    }
    
    d_Sorted[th_id] = th_id;
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
	{
		d_pos[th_id+1] =  FLT_MAX;
        d_Sorted[th_id+1] = th_id+1;
	}
}

__global__ void
projectSubMultiple16(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2, space;
   if(tx == 0)
   {
	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0; 
	   mPoint2 = 0.0;
	   space = interval/4;
   }
    __syncthreads();
    
   if(id0 == 0){
	if(d_cen[th_id].y <= mPoint1-space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].y <= mPoint1-space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].y <= mPoint1-space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1 && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1 && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1 && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].y > mPoint1-space && d_cen[th_id].y <= mPoint1 && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].y > mPoint1 && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else if(d_cen[th_id].y > mPoint1 && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 9*interval; 
	else if(d_cen[th_id].y > mPoint1 && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 10*interval;
	else if(d_cen[th_id].y > mPoint1 && d_cen[th_id].y <= mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 11*interval; 
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 12*interval;
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 13*interval;
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 14*interval;
	else if(d_cen[th_id].y > mPoint1+space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 15*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }

   if(id0 == 1){
	if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 9*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 10*interval;
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].z > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 11*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 12*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z > mPoint2-space && d_cen[th_id].z <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 13*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z > mPoint2 && d_cen[th_id].z <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 14*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].z > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 15*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }
    
   if(id0 == 2){
	if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].y > mPoint2 && d_cen[th_id].y <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 2*interval;
	else if(d_cen[th_id].x <= mPoint1-space && d_cen[th_id].y > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 3*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].y <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 4*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 5*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].y > mPoint2 && d_cen[th_id].y <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 6*interval; 
	else if(d_cen[th_id].x > mPoint1-space && d_cen[th_id].x <= mPoint1 && d_cen[th_id].y > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 7*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 8*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 9*interval; 
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y > mPoint2 && d_cen[th_id].y <= mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 10*interval;
	else if(d_cen[th_id].x > mPoint1 && d_cen[th_id].x <= mPoint1+space && d_cen[th_id].y > mPoint2+space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 11*interval; 
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y <= mPoint2-space)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 12*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y > mPoint2-space && d_cen[th_id].y <= mPoint2)
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 13*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y > mPoint2 && d_cen[th_id].y <= mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 14*interval;
	else if(d_cen[th_id].x > mPoint1+space && d_cen[th_id].y > mPoint2+space)
	    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w + 15*interval; 
	else
		d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w; 
    }
    
    d_Sorted[th_id] = th_id;
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
	{
		d_pos[th_id+1] =  FLT_MAX;
        d_Sorted[th_id+1] = th_id+1;
	}
}

__global__ void
project(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, udword* d_Sorted, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
	__shared__ float3 PCA; 
   if(tx == 0)
   {
	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
   }
    __syncthreads();

    d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
	d_Sorted[th_id] = th_id;
     if(th_id == size-1) 
	{
		d_pos[th_id+1] =  FLT_MAX;
        d_Sorted[th_id+1] = th_id+1;
	}
}

__global__ void
setRelevantSphere(float4 *d_cen, float *d_pos, udword* d_thread, udword* d_Sorted, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   float Pos = d_pos[th_id]; 
   if(Pos != 0xffffffff)
   {
       d_pos[d_thread[th_id]-1] = Pos;
       //d_cen[d_thread[th_id]-1] = d_cen[th_id]; 
       d_Sorted[d_thread[th_id]-1] = th_id;//d_Sorted[th_id];
   } 
       
    if(th_id == size-1) d_pos[d_thread[th_id]] =  FLT_MAX; 
}

__global__ void
cullCrossSphere(float4 *d_cen, float *d_pos, float* d_matrix, int* d_lamda, udword* d_thread, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   //__shared__ float3 PCA; 
   __shared__ int id;
   __shared__ float mPoint;
   if(tx == 0)
   {
	   id = d_lamda[1];
	   mPoint = 0.0;   //mPoint = d_matrix[id]/size;  
   }
    __syncthreads();
    
   float r; 
   if(id == 0){
		r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint && d_cen[th_id].x - r <= mPoint)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }

   if(id == 1){
        r = d_cen[th_id].w;
		if(d_cen[th_id].y + r >= mPoint && d_cen[th_id].y - r <= mPoint)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}		
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }
    
   if(id == 2){
        r = d_cen[th_id].w;
		if(d_cen[th_id].z + r >= mPoint && d_cen[th_id].z - r <= mPoint)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
}

__global__ void
cullCrossSphereMultiple(float4 *d_cen, float *d_pos, float* d_matrix, int* d_lamda, udword* d_thread, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   //__shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2;
   if(tx == 0)
   {
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0;   
	   mPoint2 = 0.0;
   }
    __syncthreads();
    
   float r; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		if(d_cen[th_id].y + r >= mPoint1 && d_cen[th_id].y - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 && d_cen[th_id].z - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1 && d_cen[th_id].x - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}		
		else if(d_cen[th_id].z + r >= mPoint2 && d_cen[th_id].z - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1 && d_cen[th_id].x - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2 && d_cen[th_id].y - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
}

__global__ void
cullCrossSphereMultiple9(float4 *d_cen, float *d_pos, float* d_matrix, int* d_lamda, udword* d_thread, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   //__shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2, space;
   if(tx == 0)
   {
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0; 
	   mPoint2 = 0.0;
	   space = interval/6;
   }
   __syncthreads();

   float r; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		if(d_cen[th_id].y + r >= mPoint1-space && d_cen[th_id].y - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1+space && d_cen[th_id].y - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space && d_cen[th_id].z - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space && d_cen[th_id].z - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1-space && d_cen[th_id].x - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space && d_cen[th_id].x - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space && d_cen[th_id].z - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space && d_cen[th_id].z - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1-space && d_cen[th_id].x - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space && d_cen[th_id].x - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2-space && d_cen[th_id].y - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2+space && d_cen[th_id].y - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
}

__global__ void
cullCrossSphereMultiple16(float4 *d_cen, float *d_pos, float* d_matrix, int* d_lamda, udword* d_thread, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   //__shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float mPoint1, mPoint2, space;
   if(tx == 0)
   {
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   mPoint1 = 0.0; 
	   mPoint2 = 0.0;
	   space = interval/4;
   }
   __syncthreads();

   float r; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		if(d_cen[th_id].y + r >= mPoint1 && d_cen[th_id].y - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 && d_cen[th_id].z - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1-space && d_cen[th_id].y - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1+space && d_cen[th_id].y - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space && d_cen[th_id].z - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space && d_cen[th_id].z - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1 && d_cen[th_id].x - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 && d_cen[th_id].z - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1-space && d_cen[th_id].x - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space && d_cen[th_id].x - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space && d_cen[th_id].z - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space && d_cen[th_id].z - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
		if(d_cen[th_id].x + r >= mPoint1 && d_cen[th_id].x - r <= mPoint1)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2 && d_cen[th_id].y - r <= mPoint2)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1-space && d_cen[th_id].x - r <= mPoint1-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space && d_cen[th_id].x - r <= mPoint1+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2-space && d_cen[th_id].y - r <= mPoint2-space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2+space && d_cen[th_id].y - r <= mPoint2+space)
		{
			d_pos[th_id] = r;
			//d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0;
			//d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
}

__global__ void
cullRelevantSphere(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, float* d_matrix, udword* d_Sorted, int* d_lamda, udword* d_thread, float* d_radii, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id;
   __shared__ float maxR, mPoint;
   if(tx == 0)
   {
   	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id = d_lamda[1];
	   maxR = *d_radii; 
	   mPoint = 0.0;  
   }
    __syncthreads();
    
   d_Sorted[th_id] = th_id; 
   float r;//, maxR; 
   if(id == 0){
		r = d_cen[th_id].w;
		//maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint - maxR && d_cen[th_id].x - r <= mPoint + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }

   if(id == 1){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].y + r >= mPoint - maxR && d_cen[th_id].y - r <= mPoint + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}		
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }
    
   if(id == 2){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].z + r >= mPoint - maxR && d_cen[th_id].z - r <= mPoint + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
   {
      d_pos[th_id+1] =  FLT_MAX; 
      d_Sorted[th_id+1] = th_id + 1;
   }
}

__global__ void
cullRelevantSphereMultiple(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, float* d_matrix, udword* d_Sorted, int* d_lamda, udword* d_thread, float* d_radii, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float maxR, mPoint1, mPoint2;
   if(tx == 0)
   {
   	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   maxR = *d_radii; 
	   mPoint1 = 0.0;  
	   mPoint2 = 0.0;  
   }
    __syncthreads();
    
   d_Sorted[th_id] = th_id; 
   float r;//, maxR; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		//maxR = *d_radii;
		if(d_cen[th_id].y + r >= mPoint1 - maxR && d_cen[th_id].y - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 - maxR && d_cen[th_id].z - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1 - maxR && d_cen[th_id].x - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}		
		else if(d_cen[th_id].z + r >= mPoint2 - maxR && d_cen[th_id].z - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1 - maxR && d_cen[th_id].x - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2 - maxR && d_cen[th_id].y - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
   {
      d_pos[th_id+1] =  FLT_MAX; 
      d_Sorted[th_id+1] = th_id + 1;
   }
}

__global__ void
cullRelevantSphereMultiple9(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, float* d_matrix, udword* d_Sorted, int* d_lamda, udword* d_thread, float* d_radii, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float maxR, mPoint1, mPoint2, space;
   if(tx == 0)
   {
   	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   maxR = *d_radii; 
	   mPoint1 = 0.0;  
	   mPoint2 = 0.0;  
	   space = interval/6;
   }
    __syncthreads();
    
   d_Sorted[th_id] = th_id; 
   float r;//, maxR; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		//maxR = *d_radii;
		if(d_cen[th_id].y + r >= mPoint1-space - maxR && d_cen[th_id].y - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1+space - maxR && d_cen[th_id].y - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space - maxR && d_cen[th_id].z - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space - maxR && d_cen[th_id].z - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1-space - maxR && d_cen[th_id].x - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space - maxR && d_cen[th_id].x - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space - maxR && d_cen[th_id].z - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space - maxR && d_cen[th_id].z - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1-space - maxR && d_cen[th_id].x - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space - maxR && d_cen[th_id].x - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2-space - maxR && d_cen[th_id].y - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2+space - maxR && d_cen[th_id].y - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
   {
      d_pos[th_id+1] =  FLT_MAX; 
      d_Sorted[th_id+1] = th_id + 1;
   }
}

__global__ void
cullRelevantSphereMultiple16(float *g_idata1, float *g_idata2, float4 *d_cen, float *d_pos, float* d_matrix, udword* d_Sorted, int* d_lamda, udword* d_thread, float* d_radii, int size, float interval)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   __shared__ float3 PCA; 
   __shared__ int id0, id1, id2;
   __shared__ float maxR, mPoint1, mPoint2, space;
   if(tx == 0)
   {
   	   PCA.x = g_idata1[0];
	   PCA.y = g_idata1[1];
	   PCA.z = g_idata1[2]; 
	   id0 = d_lamda[0];
	   id1 = d_lamda[1];
	   id2 = d_lamda[2];
	   maxR = *d_radii; 
	   mPoint1 = 0.0;  
	   mPoint2 = 0.0;  
	   space = interval/4;
   }
    __syncthreads();
    
   d_Sorted[th_id] = th_id; 
   float r;//, maxR; 
   if(id0 == 0){
		r = d_cen[th_id].w;
		//maxR = *d_radii;
		if(d_cen[th_id].y + r >= mPoint1 - maxR && d_cen[th_id].y - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 - maxR && d_cen[th_id].z - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1-space - maxR && d_cen[th_id].y - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint1+space - maxR && d_cen[th_id].y - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space - maxR && d_cen[th_id].z - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space - maxR && d_cen[th_id].z - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }

   if(id0 == 1){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1 - maxR && d_cen[th_id].x - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2 - maxR && d_cen[th_id].z - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1-space - maxR && d_cen[th_id].x - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space - maxR && d_cen[th_id].x - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2-space - maxR && d_cen[th_id].z - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].z + r >= mPoint2+space - maxR && d_cen[th_id].z - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }
    
   if(id0 == 2){
        r = d_cen[th_id].w;
        //maxR = *d_radii;
		if(d_cen[th_id].x + r >= mPoint1 - maxR && d_cen[th_id].x - r <= mPoint1 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2 - maxR && d_cen[th_id].y - r <= mPoint2 + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1-space - maxR && d_cen[th_id].x - r <= mPoint1-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].x + r >= mPoint1+space - maxR && d_cen[th_id].x - r <= mPoint1+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2-space - maxR && d_cen[th_id].y - r <= mPoint2-space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else if(d_cen[th_id].y + r >= mPoint2+space - maxR && d_cen[th_id].y - r <= mPoint2+space + maxR)
		{
			d_pos[th_id] = PCA.x * d_cen[th_id].x + PCA.y * d_cen[th_id].y + PCA.z * d_cen[th_id].z - d_cen[th_id].w;
			d_thread[th_id] = 1;
		}
		else
		{
			d_pos[th_id] = 0xffffffff;
			d_thread[th_id] = 0;
		}
    }  
    
    //d_pos[th_id] = d_cen[th_id].x - d_cen[th_id].w;
    if(th_id == size-1) 
   {
      d_pos[th_id+1] =  FLT_MAX; 
      d_Sorted[th_id+1] = th_id + 1;
   }
}

__global__ void
calDev(float4 *g_idata1, float4 *g_idata2, float4 *g_odata1, float4 *g_odata2, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

    float C00, C01, C02, C11, C12, C22;
    float4 temp = g_idata2[th_id] - g_idata1[0]/size;   
    C00 = temp.x*temp.x/size; 
    C01 = temp.x*temp.y/size; 
    C02 = temp.x*temp.z/size;
    C11 = temp.y*temp.y/size;
    C12 = temp.y*temp.z/size;
    C22 = temp.z*temp.z/size;    
     
    g_odata1[th_id] = make_float4(C00, C01, C02, 0.0);
    g_odata2[th_id] = make_float4(C11, C12, C22, 0.0); 
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(float4 *g_idata, float4 *g_odata, unsigned int n)
{
    extern __shared__ float4 sdata[];
    
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = make_float4(0.0, 0.0, 0.0, 0.0);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        sdata[tid] += g_idata[i];
        
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            sdata[tid] += g_idata[i+blockSize];  
        i += gridSize;
    } 
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
    if(tid < 32) {
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32]; 
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16]; 
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8]; 
		if (blockSize >=   8) sdata[tid] += sdata[tid +  4]; 
		if (blockSize >=   4) sdata[tid] += sdata[tid +  2]; 
		if (blockSize >=   2) sdata[tid] += sdata[tid +  1]; 
    }
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void 
assignThreadIDSphere(udword* d_thread, udword* id_thread, udword nbox)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     udword loopNum;
     udword step = 0;
     if (th_id < nbox)
     {
		 loopNum = d_thread[th_id+1] - d_thread[th_id]; // th_id: [0, nbox-1]
		 udword offset = th_id ==0 ? 0 : d_thread[th_id];
		 for (udword i = 0; i < loopNum; i++)
		 {
			id_thread[offset + i] = th_id;             
		 }
		  if (th_id == nbox-1) id_thread[d_thread[th_id+1]]=nbox;
      }
}

__global__ void 
CounterThreadSphPCATexSub(float* d_cen, float* d_pos, udword* d_thread, udword* d_Sorted, udword asig_num, udword nbox, float* d_mean, int* d_lamda, float interval)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     if (th_id < nbox)
     {
		 //udword index_cur = d_Sorted[th_id];
		 //float4 cen0 = tex1Dfetch(cenTex, index_cur); 
		 float4 cen0 = tex1Dfetch(cenTex, th_id); 
		 //float cen_cur = cen0.x;
		 float max_cur = d_pos[th_id] + 2*cen0.w;
		 udword mid, starts, ends;
		 udword low, up;
		 starts = th_id;
		 ends = nbox;
		 float midvalue;
		 d_thread[th_id] = 1;
	     
		 //if (max_cur > d_pos[nbox-1]) // [ai,bi]; bi > max(ai)
		 if (max_cur > tex1Dfetch(posTex, nbox-1))
		 {
			   d_thread[th_id] = (nbox -1 - th_id)/asig_num+1;
		 }
		 else
		 {
			 while(starts < ends) // O(logn)
			 {
				 mid = (starts + ends) / 2;
				 low = mid-1 > 0 ? mid-1 : 0;
				 up = mid+1 < nbox ? mid+1 : nbox-1; 
				 midvalue = d_pos[mid];

				if (max_cur <= tex1Dfetch(posTex, up) && max_cur >= midvalue) 
				 {
					d_thread[th_id] = (mid - th_id)/asig_num+1;
					break; 
				 }
				 else
				 {
					if (max_cur < midvalue)
					{
						ends = mid;
					}
					else
					{
						starts = mid + 1;
					}
				 }
			 }
		 }
		 if (th_id == nbox-1) d_thread[nbox] = d_thread[th_id];
     }
}

__global__ void 
reorderSphere(udword* d_Sorted, float* d_cen, float4* sortedCen, udword nbox)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     if (th_id < nbox)
     {
		udword index = d_Sorted[th_id];//tex1Dfetch(sortTex, th_id);
		sortedCen[th_id] = tex1Dfetch(cenTex, index); 
     }
}

__global__ void 
calBoxHashSub32(float* d_cen, int* d_hash, float* d_mean, float gridsize, udword nbox, int* d_lamda, float* d_matrix)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     float ymin, ymax, zmin, zmax;
     unsigned char ys1, ys2, zs1, zs2;
     ys1 = ys2 = 0;
     zs1 = zs2 = 0;
     __shared__ int id0, id1;
     __shared__ float meanY, meanZ;
     if(tx == 0)
     {
	     id0 = d_lamda[2];
         id1 = d_lamda[1];
         meanY = 0.0;//d_matrix[id0]/nbox;  
		 meanZ = 0.0;//d_matrix[id1]/nbox;
     }
     __syncthreads();
   
     if (th_id < nbox)
     {
		 float4 cen = tex1Dfetch(cenTex, th_id); 
		 if(id0 ==0)
		 {
		   ymin = cen.x - cen.w;
		   ymax = cen.x + cen.w;
		 }
		 else if (id0 ==1)
		 {
		   ymin = cen.y - cen.w;
		   ymax = cen.y + cen.w;
		 }
		 else
		 {
		   ymin = cen.z - cen.w;
		   ymax = cen.z + cen.w;
		 }
	     
		 if(id1 ==0)
		 {
		   zmin = cen.x - cen.w;
		   zmax = cen.x + cen.w;
		 }
		 else if (id1 ==1)
		 {
		   zmin = cen.y - cen.w;
		   zmax = cen.y + cen.w;
		 }
		 else
		 {
		   zmin = cen.z - cen.w;
		   zmax = cen.z + cen.w;
		 }
		 //ymin = cen.y - cen.w;
		 //ymax = cen.y + cen.w;
		 //zmin = cen.z - cen.w;
		 //zmax = cen.z + cen.w;	
		 float width = gridsize/2;
		 
		 if (ymin <= meanY - 3*width)
			ys1 = 0;
		 else if (ymin > meanY - 3*width && ymin <= meanY - 2*width)
			ys1 = 1;
		 else if (ymin > meanY - 2*width && ymin <= meanY - width)
			ys1 = 2;
		 else if (ymin > meanY - width && ymin <= meanY)
			ys1 = 3;
		 else if (ymin > meanY && ymin <=meanY + width)
			ys1 = 4;
		 else if (ymin > meanY + width && ymin <= meanY + 2*width)
			ys1 = 5;
		 else if (ymin > meanY + 2*width && ymin <= meanY + 3*width)
			ys1 = 6;
		 else if (ymin > meanY + 3*width)
			ys1 = 7;
		    
		 if (ymax <= meanY - 3*width)
			ys2 = 0;
		 else if (ymax > meanY - 3*width && ymax <= meanY - 2*width)
			ys2 = 1;
		 else if (ymax > meanY - 2*width && ymax <= meanY - width)
			ys2 = 2;
		 else if (ymax > meanY - width && ymax <= meanY)
			ys2 = 3;
		 else if (ymax > meanY && ymax <=meanY + width)
			ys2 = 4;
		 else if (ymax > meanY + width && ymax <= meanY + 2*width)
			ys2 = 5;
		 else if (ymax > meanY + 2*width && ymax <= meanY + 3*width)
			ys2 = 6;
		 else if (ymax > meanY + 3*width)
			ys2 = 7;
		    
		 if (zmin <= meanZ - gridsize)
			zs1 = 0;
		 else if (zmin > meanZ - gridsize && zmin <= meanZ)
			zs1 = 1;
		 else if (zmin > meanZ && zmin <= meanZ + gridsize)
			zs1 = 2;
		 else
			zs1 = 3;
		    
		 if (zmax <= meanZ - gridsize)
			zs2 = 0;
		 else if (zmax > meanZ - gridsize && zmax <= meanZ)
			zs2 = 1;
		 else if (zmax > meanZ && zmax <= meanZ + gridsize)
			zs2 = 2;
		 else
			zs2 = 3;
		 
		 int bit = 0;
		 int result = 0;
		 for(int i = ys1; i<=ys2; i++)
			 for(int j = zs1; j<=zs2; j++)
				 {
						bit = i <<2 | j;
						result |= 1 << bit;
				 } 
		 d_hash[th_id] = result;
     }
}

__global__ void 
ZeroSetSphere(udword* d_pairs, udword n)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     if (th_id < n)
     d_pairs[th_id] = 0;
}

__global__ void 
assemPairsSphereSub(udword* d_pairs, udword* d_mulpairs, udword* d_Sorted, udword* d_thread, udword nbox, udword totalThreadNum, udword asig_num, udword overlapnum, udword rbox, udword line)
{
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     udword loopNum;
     if (th_id < nbox)
     {
		 loopNum = d_thread[th_id+1] - d_thread[th_id]; // th_id: [0, nbox-1]
		 udword subID = d_thread[th_id];
		 udword overlapID;
		 udword index0 = d_Sorted[th_id];
		 //d_pairs[th_id] = *index0;
		 udword step = 0;
		 udword offsetPar;
		 //offsetPar = rbox + d_mulpairs[subID];
		 offsetPar = rbox + (subID -1 < 0 ? 0 : d_mulpairs[subID-1]);
		 udword id = subID - 1 < 0 ? 0 : subID - 1;
		 udword i;
		 for (i = 0; i < loopNum; i++)
		 {
			d_mulpairs[id+i] = 0;
			for (udword offset = 1; offset < asig_num+1; offset++) // offset < asig_num, store asig_num pairs
			{ 
			   overlapID = d_mulpairs[subID+i + offset*totalThreadNum];
			   //d_mulpairs[subID+i] = 0;
			   d_mulpairs[subID+i + offset*totalThreadNum] = 0; //Dec.29th, reset d_mulpairs array
			   if (overlapID)
			   {
				 d_pairs[offsetPar + step] = index0+1;    
				 d_pairs[offsetPar + step + overlapnum/2 * line] = overlapID; 
				step++;  
				}
			   else{
				 break;
				}
			}
		 }
		 if(th_id == nbox-1) d_mulpairs[subID+i-1] = 0;
     } // if (th_id < nbox)
     
}

__global__ void 
kernel_SphereTexCullPCASub(float* d_cen, float* d_pos, udword* d_Sorted, udword* d_pairs, udword* d_thread, udword* id_thread, 
                                    udword* d_mulpairs, udword totalThreadNum, udword nbox, udword overlapnum, udword asig_num, float* d_mean)
{
     __shared__ float4 pairsCen[512]; 
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     //udword maxobjID = id_thread[512*blockIdx.x + 512];
     udword minobjID = id_thread[512*blockIdx.x];
     udword* start_index = &d_Sorted[minobjID];
     udword shareSize = 1;
     //float4 cen_0;//, extent_0;//, cen_1, extent_1;
     
     if (th_id < totalThreadNum)
     {
		 if (minobjID + 512 < nbox)
         {
			 //cen_0 = tex1Dfetch(cenTex, minobjID+tid); 
			 pairsCen[tid] = tex1Dfetch(cenTex, minobjID+tid);//make_float4(cen_0.x, cen_0.y, cen_0.z, cen_0.w);
			 shareSize = 512; 
		 }
		 else if(tid < nbox - minobjID)
		 {
		     //cen_0 = tex1Dfetch(cenTex, minobjID+tid); 
			 pairsCen[tid] = tex1Dfetch(cenTex, minobjID+tid);//make_float4(cen_0.x, cen_0.y, cen_0.z, cen_0.w);
			 shareSize = nbox - minobjID;
	     }
     }
     __syncthreads();
     
     if (th_id < totalThreadNum)
     {
		 udword  index0;
		 udword  index;
		 udword objID = id_thread[th_id];
         
	     bool pass = true;
	     
         index0 = objID - minobjID;
         float4 cen0 = pairsCen[index0];
                           
		 if (objID!=id_thread[th_id+1]) // obj1:[0,0,0], obj2:[1,1,1], objn-1:[n-1,n-1], objn:[n], n=box
		 {
			pass = false;
		}

		udword starts = objID == 0? 0 : d_thread[objID];
		udword cur_id= th_id - starts;
		index = cur_id == 0 ? index0 + 1 : index0 + cur_id * asig_num;
		udword cur_obj =  cur_id == 0 ? objID+1 : objID + cur_id * asig_num;
				
		udword offset = 1;//cur_id == 0 ? 1 : 0;
		//udword offset = 2;// for result pairs compact
        udword i;
        float4 cen;
        //udword indexBits = 0;
        //udword mask = 1;
        float t1, e1, x0, y0, z0, r0;
        int bits, bitsTemp;
        bits = tex1Dfetch(hashTex, objID);//d_hash[objID];
        //float dPos = d_pos[cur_obj];
        //float distance = dPos + 2*cen0.w;
        
		 i = cur_id == 0 ? 1 : 0;
		 // #pragma unroll 64 
		 while(i<asig_num) // the overlap candidates are as many as the boxed bounding volume
		 {
			 //if(!pass && dPos > distance)
			 //break; 
			 
			 bitsTemp =  tex1Dfetch(hashTex, index - index0 + objID);
			 if (bits & bitsTemp)//(bits & d_hash[index - index0 + objID])	 
			{ 
			 if (index > shareSize-1)
			 {
				 //cen = tex1Dfetch(cenTex, *(index - index0 + &d_Sorted[objID])); //make_float4(cenTemp.x, cenTemp.y, cenTemp.z, cenTemp.w);
				 cen = tex1Dfetch(cenTex, index - index0 + objID);
			 }
			 else
			 {
				 cen = pairsCen[index];
			 }
			 x0 = cen0.x-cen.x;
			 y0 = cen0.y-cen.y;
			 z0 = cen0.z-cen.z;
			 r0 =  cen0.w+cen.w;
			 t1 = x0*x0 + y0*y0 + z0*z0;
			 e1 = r0*r0;	 
			 if(t1 < e1) //<=
			 {
			      //indexBits = indexBits | mask << offset; 
				  d_mulpairs[th_id+offset*totalThreadNum] = *(start_index + index) + 1; //
				  offset++;
			 } 
			} //if (bits & d_hash[index - index0 + objID]) 
			 index++;
	         i++;
			 //if(!pass) cur_obj++;
		 }  			 
		 
		 d_mulpairs[th_id] = offset - 1;// for results compact

		 //__syncthreads();
         //d_mulpairs[th_id] = indexBits;
		 
     } //if (th_id < totalThreadNum)
}

__global__ void 
kernel_SphereTexCullPCAElong(float* d_cen, float* d_pos, udword* d_Sorted, udword* d_pairs, udword* d_thread, udword* id_thread, 
                                    udword* d_mulpairs, udword totalThreadNum, udword nbox, udword overlapnum, udword asig_num, float* d_mean)
{
     __shared__ float4 pairsCen[512]; 
     udword tx = threadIdx.x;
     udword ty = threadIdx.y;
     udword bw = blockDim.x;
     udword bh = blockDim.y;
     udword tid = ty * bw + tx;//__umul24(ty, bw) + tx;
     udword bid = blockIdx.x * bw * bh;//__umul24(__umul24(blockIdx.x, bw), bh);
     udword th_id =  bid + tid;
     //udword maxobjID = id_thread[512*blockIdx.x + 512];
     udword minobjID = id_thread[512*blockIdx.x];
     udword* start_index = &d_Sorted[minobjID];
     udword shareSize = 1;
     //float4 cen_0;//, extent_0;//, cen_1, extent_1;
     
     if (th_id < totalThreadNum)
     {
		 if (minobjID + 512 < nbox)
         {
			 //cen_0 = tex1Dfetch(cenTex, minobjID+tid); 
			 pairsCen[tid] = tex1Dfetch(cenTex, minobjID+tid);//make_float4(cen_0.x, cen_0.y, cen_0.z, cen_0.w);
			 shareSize = 512; 
		 }
		 else if(tid < nbox - minobjID)
		 {
		     //cen_0 = tex1Dfetch(cenTex, minobjID+tid); 
			 pairsCen[tid] = tex1Dfetch(cenTex, minobjID+tid);//make_float4(cen_0.x, cen_0.y, cen_0.z, cen_0.w);
			 shareSize = nbox - minobjID;
	     }
     }
     __syncthreads();
     
     if (th_id < totalThreadNum)
     {
		 udword  index0;
		 udword  index;
		 udword objID = id_thread[th_id];
         
	     bool pass = true;
	     
         index0 = objID - minobjID;
         float4 cen0 = pairsCen[index0];
                           
		 if (objID!=id_thread[th_id+1]) // obj1:[0,0,0], obj2:[1,1,1], objn-1:[n-1,n-1], objn:[n], n=box
		 {
			pass = false;
		}

		udword starts = objID == 0? 0 : d_thread[objID];
		udword cur_id= th_id - starts;
		index = cur_id == 0 ? index0 + 1 : index0 + cur_id * asig_num;
		udword cur_obj =  cur_id == 0 ? objID+1 : objID + cur_id * asig_num;
				
		udword offset = 1;//cur_id == 0 ? 1 : 0;
		//udword offset = 2;// for result pairs compact
        udword i;
        float4 cen;
        //udword indexBits = 0;
        //udword mask = 1;
        float t1, e1, x0, y0, z0, r0;
        int bits, bitsTemp;
        bits = tex1Dfetch(hashTex, objID);//d_hash[objID];
        float dPos = d_pos[objID];
        float distance = dPos + 2*cen0.w;
        
		 i = cur_id == 0 ? 1 : 0;
		 // #pragma unroll 64 
		 while(i<asig_num) // the overlap candidates are as many as the boxed bounding volume
		 {
			 if(!pass && d_pos[cur_obj] > distance)
			 break; 
			 
			 bitsTemp =  tex1Dfetch(hashTex, index - index0 + objID);
			 if (bits & bitsTemp)//(bits & d_hash[index - index0 + objID])	 
			{ 
			 if (index > shareSize-1)
			 {
				 //cen = tex1Dfetch(cenTex, *(index - index0 + &d_Sorted[objID])); //make_float4(cenTemp.x, cenTemp.y, cenTemp.z, cenTemp.w);
				 cen = tex1Dfetch(cenTex, index - index0 + objID);
			 }
			 else
			 {
				 cen = pairsCen[index];
			 }
			 x0 = cen0.x-cen.x;
			 y0 = cen0.y-cen.y;
			 z0 = cen0.z-cen.z;
			 r0 =  cen0.w+cen.w;
			 t1 = x0*x0 + y0*y0 + z0*z0;
			 e1 = r0*r0;	 
			 if(t1 < e1) //<=
			 {
			      //indexBits = indexBits | mask << offset; 
				  d_mulpairs[th_id+offset*totalThreadNum] = *(start_index + index)+1; //
				  offset++;
			 } 
			} //if (bits & d_hash[index - index0 + objID]) 
			 index++;
	         i++;
			 if(!pass) cur_obj++;
		 }  			 
		 
		 d_mulpairs[th_id] = offset - 1;// for results compact

		 //__syncthreads();
         //d_mulpairs[th_id] = indexBits;
		 
     } //if (th_id < totalThreadNum)
}

__global__ void
recover(udword* g_idata1, udword* g_idata2, udword size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id == 0) g_idata2[th_id] = 1;
	if(th_id > 0 && th_id < size)
	{
		if(g_idata1[th_id-1] != g_idata1[th_id])
			g_idata2[th_id] = 1;
		else
			g_idata2[th_id] = 0;
		//g_idata2[th_id + 2*size] = g_idata1[th_id + overlapNum/2 * objNum];
	}
}

__global__ void
reverseIndex(udword* g_idata1, udword* g_idata2, udword size)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos; 
    if(th_id == 0) 
	{
		Pos = g_idata1[th_id]; 
		g_idata2[Pos-1] = th_id;
	}
    if(th_id > 0 && th_id < size)
	{
		Pos = g_idata1[th_id]; 
		if(Pos != g_idata1[th_id-1])
			g_idata2[Pos-1] = th_id;
	}
}

//addPairs<<<grid1, threads>>>(d_thread, d_pairs, d_mulpairs, objNum, realpairsNum, asig_num);
__global__ void
addPairs(udword* g_idata1, udword* g_idata2, udword* g_idata3, udword objNum, uint pairsNum, uint asig_num)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < objNum)
	{
		int pairID = g_idata1[th_id];
		int objID = 0;
		int step = 1;
		if(pairID != 0xffffffff)
		{
			objID = g_idata2[pairID];
			step = g_idata3[objID]+1;
			do
			{
				g_idata3[objID + step*objNum] = g_idata2[pairID + pairsNum];
				step++;
				if(step > asig_num-1) break;
			}
			while(pairID < pairsNum && objID == g_idata2[++pairID]);
		}
	}
}

__global__ void
setPairsIndex(udword* g_idata1, udword* g_idata2, udword size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

    udword Pos; 
    if(th_id == 0) g_idata2[0] = 0;
    if(th_id > 0 && th_id < size)
	{
		Pos = g_idata1[th_id]; 
		if(Pos != g_idata1[th_id-1])
			g_idata2[Pos-1] = th_id;
	}
}

__global__ void
pairsMatrix1(udword* g_idata1, udword* g_idata2, udword* g_idata3, udword size, udword line, udword objNum, uint pairsNum, uint asig_num)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		int pairID = g_idata3[th_id];
		int objID = 0;
		int step = 1;
		objID = g_idata2[pairID];
		do
		{
			g_idata1[objID + step*objNum] = g_idata2[pairID + line];
			step++;
			if(step > asig_num-1) break;
		}
		while(pairID < pairsNum && objID == g_idata2[++pairID]);
        g_idata1[objID] = step-1;
	}
}

__global__ void
pairsMatrix2(udword* g_idata1, udword* g_idata2, udword* g_idata3, udword relsize, udword size1, udword line, udword objNum, uint pairsNum, uint asig_num)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < relsize)
	{
		int pairID = g_idata3[th_id + size1];
		int objID, objID1, step, isPair, num;
		objID = 0;
		objID1 = 0;
		//step = 1;
		//isPair = 1;
		num = 0;
		objID = g_idata2[pairID];
		do
		{
			step = 1;
			isPair = 1;
			objID1 = g_idata1[objID + step*objNum] ;
			while(objID1 != 0)
			{
				if(objID1!= g_idata2[pairID + line])
					step++;
				else{
					isPair = 0;
					break;
				}
				objID1 = g_idata1[objID + step*objNum] ;
			}
            
			if(step > asig_num-1) break;

			if(isPair){
				g_idata1[objID + step*objNum] = g_idata2[pairID + line];	
				num++;
			}
		}
		while(pairID < pairsNum && objID == g_idata2[++pairID]);
		g_idata1[objID] += num;
	}
}

__global__ void
outputPairs(udword* g_idata1, udword* g_idata2, udword size, udword num)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		int startID, endID, step;
		//startID = th_id == 0 ? 0 : g_idata1[th_id-1];
		//endID = g_idata1[th_id];
		step = 1;
		for(int i = startID; i < endID; i++)
		{
			g_idata2[i] = g_idata1[th_id+step*size];//???
			g_idata2[i+num] = th_id+1;
			step++;
			//if(step > 255) break;
		}
	}
}

__global__ void
duplicate(udword* g_idata1, udword* g_idata2, udword size, udword line)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		g_idata2[th_id] = g_idata1[th_id];
		g_idata2[th_id + 2*size] = g_idata1[th_id + line];
		g_idata2[th_id + size] = g_idata1[th_id + line];
		g_idata2[th_id + 3*size] = g_idata1[th_id];

		/*g_idata2[th_id + 2*size] = g_idata1[th_id];
		g_idata2[th_id + 4*size] = g_idata1[th_id + line];
		g_idata2[th_id + 3*size] = g_idata1[th_id + line];
		g_idata2[th_id + 5*size] = g_idata1[th_id];
        g_idata2[th_id + 6*size] = th_id;
		g_idata2[th_id + 7*size] = th_id + size;*/
	}
}

__global__ void
duplicateOne(udword* g_idata1, udword* g_idata2, udword size, udword line)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		g_idata2[th_id] = g_idata1[th_id + line];
		g_idata2[th_id + 2*size] = g_idata1[th_id];
		g_idata2[th_id + size] = g_idata1[th_id];
		g_idata2[th_id + 3*size] = g_idata1[th_id + line];
	}
}

__global__ void
duplicateInterleave(udword* g_idata1, udword* g_idata2, udword size, udword line)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		g_idata2[2*th_id] = g_idata1[th_id];
		g_idata2[2*th_id + 1] = g_idata1[th_id + line];
		g_idata2[2*th_id + 2*size] = g_idata1[th_id + line];
		g_idata2[2*th_id + 1 + 2*size] = g_idata1[th_id];
	}
}

__global__ void
duplicateTwo(udword* g_idata, udword size, udword stride1, udword stride2)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	int startID = stride1*size;
	int endID = stride2*size;
	if(th_id < size)
		g_idata[th_id + endID] = g_idata[th_id+startID];
}

__global__ void
reorderPairs(udword* g_idata1, udword* g_idata2, udword* g_idata3, udword size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		g_idata3[th_id] = g_idata2[g_idata1[th_id]];
		g_idata1[th_id] = th_id;
	}
}

__global__ void
orderInside(udword* g_idata1, udword* g_idata2, udword size, uint pairsNum)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	if(th_id < size)
	{
		int startID, endID, tempID, curID;
		startID = g_idata2[th_id];
		endID = th_id <size-1 ? g_idata2[th_id+1] : pairsNum;
		for(int i = startID; i < endID; i++)
		{
			tempID = g_idata1[i];
			for(int j = i+1; j < endID; j++)
			{
				curID = g_idata1[j];
				if(tempID > curID)
				{
					g_idata1[i] = curID;
					g_idata1[j] = tempID;
				}
			}
		}
	}
}

__global__ void
setSortKernel(udword* Sorted, int size)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;
    
   if(th_id < size)
	   Sorted[th_id] = th_id;
       
}

__global__ void 
reorderData(udword* Sorted, udword* data, udword* sortData, udword num)
{
     int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

     if (th_id < num)
     {
		udword index = Sorted[th_id];//tex1Dfetch(sortTex, th_id);
		sortData[th_id] = data[index]; 
     }
}

__global__ void 
scaleKernel(float4* data, udword num)
{
    int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

     if (th_id < num)
	 {
		 float4 Data = data[th_id];
		 data[th_id] = make_float4(100*Data.x, 100*Data.y, 100*Data.z, 100*Data.w); 
	 }
}

__global__ void
cullIndex(udword* g_idata1, udword* g_idata2, udword size)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos; 
    if(th_id == 0) 
		g_idata2[th_id] = 1;

    if(th_id > 0 && th_id < size)
	{
		if((g_idata1[th_id] !=  g_idata1[th_id-1]) || (g_idata1[th_id + size] !=  g_idata1[th_id-1 + size]))
			g_idata2[th_id] = 1;
	}
}

__global__ void
setCullPairs(udword* g_idata1, udword* g_idata2, udword size, udword totalPairs)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos, index; 

	if(th_id < size)
	{
		Pos = g_idata1[th_id+2*size]; 
		index = Pos - 1;

		if(th_id == 0) 
		{
			g_idata2[2*index] = g_idata1[th_id] - 1;
			g_idata2[2*index+1] = g_idata1[th_id + size] - 1;
		}
		else
		{
			if(Pos != g_idata1[th_id-1+2*size])
			{
				g_idata2[2*index] = g_idata1[th_id] - 1;
				g_idata2[2*index+1] = g_idata1[th_id + size] - 1;
			}
		}
	}

}

__global__ void
setCullPairsDirect(udword* g_idata1, udword* g_idata2, udword size)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos; 

	if(th_id < size)
	{
		g_idata2[th_id] = g_idata1[th_id] - 1;
		g_idata2[th_id + size] = g_idata1[th_id + 2*size] - 1;
	}

}

__global__ void
setPairsEntriesFlag(udword* g_idata1,  udword* g_idata2,  udword size)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos; 
	if(th_id < size)
	{
		Pos = g_idata1[2*th_id]; 

		if(th_id == 0) 
			g_idata2[th_id] = 1;
		else
		{
			if(Pos != g_idata1[2*th_id-2])
				g_idata2[th_id] = 1;
		}
	}

}

__global__ void
setPairsEntries(udword* g_idata1, udword* g_idata2, udword* g_idata3, udword size)
{
	int tx = threadIdx.x;
    int bw = blockDim.x;
    int th_id = blockIdx.x * bw + tx;//__umul24(blockIdx.x, bw) + tx;

	udword Pos; 

	if(th_id < size)
	{
		Pos = g_idata1[th_id]; 

		if(th_id == 0) 
			g_idata3[Pos-1] = th_id;//g_idata2[th_id];
		else
		{
			if(Pos != g_idata1[th_id-1])
				g_idata3[Pos-1] = th_id;//g_idata2[th_id];
		}
	}

}

extern "C"
{
uint sweepPrunSphereSubCross(float* d_cen, float* d_pos,  uint* d_Sorted,  uint* d_pairs, uint* d_thread, uint* id_thread, 
                                         uint* d_mulpairs, uint nbox, uint rbox, uint offset, uint overlapnum, uint asig_num, 
										 uint avthreadNum, float* sortedCen, int* d_hash, float* d_mean, int* d_lamda, float interval, float* d_matrix)
{
	 cutilSafeCall(cudaBindTexture(0, cenTex, d_cen, nbox*sizeof(float4)));
     cutilSafeCall(cudaBindTexture(0, posTex, d_pos, (nbox+1)*sizeof(float)));
     cutilSafeCall(cudaBindTexture(0, hashTex, d_hash, nbox*sizeof(int)));
     udword nbox_old = nbox;
     nbox = rbox;
     dim3 threads(32,16,1);//1024 threads or 32 warps for Fermi.
     dim3 grid(ceil(nbox/(32*16.0)),1,1);
     reorderSphere<<< grid, threads >>>(d_Sorted, d_cen, (float4*)sortedCen, nbox);
     cutilSafeCall(cudaUnbindTexture(cenTex));
     cutilSafeCall(cudaBindTexture(0, cenTex, sortedCen, nbox*sizeof(float4)));
     
     //calBoxHash16<<< grid, threads >>>(d_cen, d_hash, d_mean, 0.0, 0.0, 125.0, nbox); //25.0 - 100
     //calBoxHash32<<< grid, threads >>>(d_cen, d_hash, d_mean, 0.0, 0.0, interval/4, nbox); //25.0 - 100
	 calBoxHashSub32<<< grid, threads >>>(d_cen, d_hash, d_mean, interval/4, nbox, d_lamda, d_matrix);
     //nbox = rbox;
     udword totalThreadNum = 0;
     udword totalPairsNum = 0;//, lastPairsNum;
     CounterThreadSphPCATexSub<<< grid, threads >>>(d_cen, d_pos, d_thread, d_Sorted, asig_num, nbox, d_mean, d_lamda, interval);
     //CounterThreadSphPCATex<<< grid, threads >>>(d_cen, d_pos, d_thread, d_Sorted, asig_num, nbox, d_mean);
     //CounterThreadSphereTex<<< grid, threads >>>(d_cen, d_pos, d_thread, d_Sorted, asig_num, nbox); 
     //CounterThreadSphere<<< grid, threads >>>(d_cen, d_pos, d_thread, d_Sorted, asig_num, nbox); //0.48ms
      
     CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;

	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, nbox, 1, 0);
	 cudppScan(scanPlan, d_thread, d_thread, nbox+1);

	 //cudppDestroyPlan(scanPlan);	

     CUDA_SAFE_CALL( cudaMemcpy(&totalThreadNum, &(d_thread[nbox]), sizeof(udword), cudaMemcpyDeviceToHost) );
     //printf(":: %d\n", totalThreadNum);
	 //assignThreadIDSphwithoutPairs<<< grid, threads >>>(d_thread, id_thread, nbox, d_pairs);  
	 assignThreadIDSphere<<< grid, threads >>>(d_thread, id_thread, nbox);//0.2ms
	
	 udword blocksize = ceil(nbox*overlapnum/512.0);
     dim3 threads0(32,16,1);
     dim3 gridsize0(blocksize, 1, 1);
	 //ZeroSetSphere<<< gridsize0, threads0 >>>(d_pairs, nbox*overlapnum); //0.14ms, reset d_pairs array
	 
     blocksize = ceil(totalThreadNum/512.0);
     dim3 threads1(32,16,1);
     dim3 gridsize(blocksize, 1, 1);
     //cutilSafeCall( cudaMalloc( (void**)&id_thread, sizeof(udword)*(nbox*avthreadNum)) ); // cost too much time, more than 2ms
     
     if (totalThreadNum > nbox * avthreadNum)
		 printf("SubCross, Assigning too many thread! You need increase the subgroup size!......\n");
     else{
		 //cudaFuncSetCacheConfig(kernel_SphereTexCullPCASub, cudaFuncCachePreferL1);//48KB software-managed data cache cudaFuncCachePreferShared
         kernel_SphereTexCullPCASub<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num, d_mean);
         //kernel_SphereTexCullPCAPairs<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num, d_mean);
         //kernel_SphereTexCullWithPairs<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num);
         //kernel_SphereTexWithPairs<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num);
         //kernel_SphereOptimizedWithPairs<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num);
	     //kernel_SpherePosWithPairs<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num);
		 //assemPairsSphere<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, overlapnum);//0.2ms
		 //assemPairsWithoutPairsSphere<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, overlapnum);
		 /**/
		 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
		 cudppPlan(&scanPlan, config, totalThreadNum, 1, 0);
		 cudppScan(scanPlan, d_mulpairs, d_mulpairs, totalThreadNum);
		 cudppDestroyPlan(scanPlan);
		 
		 CUDA_SAFE_CALL( cudaMemcpy(&totalPairsNum, &(d_mulpairs[totalThreadNum - 1]), sizeof(udword), cudaMemcpyDeviceToHost) );
		 //printf(":: %d\n", totalPairsNum);
		 //totalPairsNum += 30;
		 //assemPairsSphereWithPairs<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, asig_num);// reset the d_mulpairs array
		 assemPairsSphereSub<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, asig_num, overlapnum, offset, nbox_old);
     }
     //cutilSafeCall(cudaFree(id_thread)); // cost too much time
     cutilSafeCall(cudaUnbindTexture(cenTex));
     cutilSafeCall(cudaUnbindTexture(posTex));
     cutilSafeCall(cudaUnbindTexture(hashTex));
     return totalPairsNum;// + 2*nbox;// + lastPairsNum;
}

uint sweepPrunSphereSubElong(float* d_cen, float* d_pos,  uint* d_Sorted,  uint* d_pairs, uint* d_thread, uint* id_thread, 
                                          uint* d_mulpairs, uint nbox, uint rbox, uint overlapnum, uint asig_num, uint avthreadNum, 
										  float* sortedCen, int* d_hash, float* d_mean, int* d_lamda, float interval, float* d_matrix)
{
	 cutilSafeCall(cudaBindTexture(0, cenTex, d_cen, nbox*sizeof(float4)));
     cutilSafeCall(cudaBindTexture(0, posTex, d_pos, (nbox+1)*sizeof(float)));
     cutilSafeCall(cudaBindTexture(0, hashTex, d_hash, nbox*sizeof(int)));
     
     dim3 threads(32,16,1);//1024 threads or 32 warps for Fermi.
     dim3 grid(ceil(nbox/(32*16.0)),1,1);
     reorderSphere<<< grid, threads >>>(d_Sorted, d_cen, (float4*)sortedCen, nbox);
     cutilSafeCall(cudaUnbindTexture(cenTex));
     cutilSafeCall(cudaBindTexture(0, cenTex, sortedCen, nbox*sizeof(float4)));
     
     //calBoxHash16<<< grid, threads >>>(d_cen, d_hash, d_mean, 0.0, 0.0, 125.0); //25.0 - 100
     //calBoxHash32<<< grid, threads >>>(d_cen, d_hash, d_mean, 0.0, 0.0, interval/4, nbox); //25.0 - 100
     calBoxHashSub32<<< grid, threads >>>(d_cen, d_hash, d_mean, interval/4, nbox, d_lamda, d_matrix);
     
     udword totalThreadNum = 0;
     udword totalPairsNum = 0;//, lastPairsNum;
     CounterThreadSphPCATexSub<<< grid, threads >>>(d_cen, d_pos, d_thread, d_Sorted, asig_num, nbox, d_mean, d_lamda, interval);
      
     CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;

	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, nbox+1, 1, 0);
	 cudppScan(scanPlan, d_thread, d_thread, nbox+1);

	 //cudppDestroyPlan(scanPlan);	

     CUDA_SAFE_CALL( cudaMemcpy(&totalThreadNum, &(d_thread[nbox]), sizeof(udword), cudaMemcpyDeviceToHost) );
     //printf(":: %d    %5d\n", totalThreadNum, nbox);
	 assignThreadIDSphere<<< grid, threads >>>(d_thread, id_thread, nbox);//0.2ms
	
	 udword blocksize = ceil(nbox*overlapnum/512.0);
     dim3 threads0(32,16,1);
     dim3 gridsize0(blocksize, 1, 1);
	 ZeroSetSphere<<< gridsize0, threads0 >>>(d_pairs, nbox*overlapnum); //0.14ms, reset d_pairs array
	 
     blocksize = ceil(totalThreadNum/512.0);
     dim3 threads1(32,16,1);
     dim3 gridsize(blocksize, 1, 1);
     //cutilSafeCall( cudaMalloc( (void**)&id_thread, sizeof(udword)*(nbox*avthreadNum)) ); // cost too much time, more than 2ms
     
     if (totalThreadNum > nbox * avthreadNum)
		 printf("Elong, Assigning too many thread! You need increase the subgroup size!......\n");
     else{
		 //cudaFuncSetCacheConfig(kernel_SphereTexCullPCAElong, cudaFuncCachePreferL1);//48KB software-managed data cache cudaFuncCachePreferShared
         kernel_SphereTexCullPCAElong<<< gridsize, threads1 >>>(d_cen, d_pos, d_Sorted, d_pairs, d_thread, id_thread, d_mulpairs, totalThreadNum, nbox, overlapnum, asig_num, d_mean);
		 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
		 cudppPlan(&scanPlan, config, totalThreadNum, 1, 0);
		 cudppScan(scanPlan, d_mulpairs, d_mulpairs, totalThreadNum);
		 cudppDestroyPlan(scanPlan);
		 
		 CUDA_SAFE_CALL( cudaMemcpy(&totalPairsNum, &(d_mulpairs[totalThreadNum - 1]), sizeof(udword), cudaMemcpyDeviceToHost) );
		 //printf(":: %d\n : %d\n", totalPairsNum, overlapnum);
		 //totalPairsNum += 30;
		 //assemPairsSphereWithPairs<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, asig_num);// reset the d_mulpairs array
		 assemPairsSphereSub<<< grid, threads >>>(d_pairs, d_mulpairs, d_Sorted, d_thread, nbox, totalThreadNum, asig_num, overlapnum, 0, nbox);
     }
     //cutilSafeCall(cudaFree(id_thread)); // cost too much time
     cutilSafeCall(cudaUnbindTexture(cenTex));
     cutilSafeCall(cudaUnbindTexture(posTex));
     cutilSafeCall(cudaUnbindTexture(hashTex));
     return totalPairsNum;// + 2*nbox;// + lastPairsNum;
}

void reduce(int size, int threads, int blocks, int whichKernel, float* d_idata, float* d_odata)
{
	dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    //int smemSize = threads * sizeof(float4);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float4) : threads * sizeof(float4);

    if (isPow2(size))
        {
            switch (threads)
            {
            case 512:
                reduce6<512, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 256:
                reduce6<256, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 128:
                reduce6<128, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 64:
                reduce6<64, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 32:
                reduce6<32, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 16:
                reduce6<16, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  8:
                reduce6<8, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  4:
                reduce6<4, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  2:
                reduce6<2, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  1:
                reduce6<1, true><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
                reduce6<512, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 256:
                reduce6<256, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 128:
                reduce6<128, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 64:
                reduce6<64, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 32:
                reduce6<32, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case 16:
                reduce6<16, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  8:
                reduce6<8, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  4:
                reduce6<4, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  2:
                reduce6<2, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            case  1:
                reduce6<1, false><<< dimGrid, dimBlock, smemSize >>>((float4*)d_idata, (float4*)d_odata, size); break;
            }
        }
      //printf("reduce complete ....");
}

void calDelta(float *g_idata1, float *g_idata2, float *g_odata1, float *g_odata2, int size)
{
	 dim3 threads(512,1);
     dim3 grid(size/512,1,1);
     calDev<<< grid, threads >>>((float4*)g_idata1, (float4*)g_idata2, (float4*)g_odata1, (float4*)g_odata2, size);
     //printf("calDelta complete ....");
}


uint optimalAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, uint* d_thread, int* d_num, float* d_radii, int size, float interval)
{
	 dim3 threads(512, 1);
     dim3 grid(size/512, 1, 1);
     float maxR;
     udword totalCrossNum, relevantNum;
	 //axisVec<<<1, 1>>>(g_idata1, g_idata2, d_lamda);
	 cullCrossSphere<<< grid, threads >>>((float4*)d_cen, d_pos, d_matrix, d_lamda, d_thread, size);
	 //compute the maximum radii
	 CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	 config.op = CUDPP_MAX;
	 config.datatype = CUDPP_FLOAT;

	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, size, 1, 0);
	 cudppScan(scanPlan, d_pos, d_pos, size);
	 CUDA_SAFE_CALL( cudaMemcpy(d_radii, &(d_pos[size-1]), sizeof(float), cudaMemcpyDeviceToDevice) );
	 //CUDA_SAFE_CALL( cudaMemcpy(&maxR, d_radii, sizeof(float), cudaMemcpyDeviceToHost) );
	 //printf("Maximum Radii %-5.6f\n", maxR);
	 	 
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;
	 cudppPlan(&scanPlan, config, size, 1, 0);
	 //cudppScan(scanPlan, d_thread, d_thread, size);
	 //CUDA_SAFE_CALL( cudaMemcpy(d_num, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToDevice) );
	 //CUDA_SAFE_CALL( cudaMemcpy(&totalCrossNum, d_num, sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("Cross num %d\n", totalCrossNum); // < 2100	
	 
	 cullRelevantSphere<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_matrix, d_Sorted, d_lamda, d_thread, d_radii, size, interval);

	 cudppScan(scanPlan, d_thread, d_thread, size);
	 //CUDA_SAFE_CALL( cudaMemcpy(d_num, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToDevice) );
	 CUDA_SAFE_CALL( cudaMemcpy(&relevantNum, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("relevant num %d\n", relevantNum); // < 15200
	 
	 setRelevantSphere<<< grid, threads >>>( (float4*)d_cen, d_pos, d_thread, d_Sorted, size);
     //projectSub<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_matrix, d_lamda, size, interval);
     
     cudppDestroyPlan(scanPlan);
     return relevantNum;
}

uint optimalAxisMultiple(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, uint* d_thread, int* d_num, float* d_radii, int size, float interval)
{
	 dim3 threads(512, 1);
     dim3 grid(size/512, 1, 1);
     float maxR;
     udword totalCrossNum, relevantNum;
	 //axisVec<<<1, 1>>>(g_idata1, g_idata2, d_lamda);
	 cullCrossSphereMultiple<<< grid, threads >>>((float4*)d_cen, d_pos, d_matrix, d_lamda, d_thread, size);
	 //cullCrossSphereMultiple9<<< grid, threads >>>((float4*)d_cen, d_pos, d_matrix, d_lamda, d_thread, size, interval);
	 //compute the maximum radii
	 CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	 config.op = CUDPP_MAX;
	 config.datatype = CUDPP_FLOAT;

	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, size, 1, 0);
	 cudppScan(scanPlan, d_pos, d_pos, size);
	 CUDA_SAFE_CALL( cudaMemcpy(d_radii, &(d_pos[size-1]), sizeof(float), cudaMemcpyDeviceToDevice) );
	 //CUDA_SAFE_CALL( cudaMemcpy(&maxR, d_radii, sizeof(float), cudaMemcpyDeviceToHost) );
	 //printf("Maximum Radii %-5.6f\n", maxR);
	 	 
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;
	 cudppPlan(&scanPlan, config, size, 1, 0);
	 //cudppScan(scanPlan, d_thread, d_thread, size);
	 //CUDA_SAFE_CALL( cudaMemcpy(d_num, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToDevice) );
	 //CUDA_SAFE_CALL( cudaMemcpy(&totalCrossNum, d_num, sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("Cross num %d\n", totalCrossNum); // < 2100	

	 cullRelevantSphereMultiple<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_matrix, d_Sorted, d_lamda, d_thread, d_radii, size, interval);
	 //cullRelevantSphereMultiple9<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_matrix, d_Sorted, d_lamda, d_thread, d_radii, size, interval);

	 cudppScan(scanPlan, d_thread, d_thread, size);
	 //CUDA_SAFE_CALL( cudaMemcpy(d_num, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToDevice) );
	 CUDA_SAFE_CALL( cudaMemcpy(&relevantNum, &(d_thread[size-1]), sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("relevant num %d\n", relevantNum); // < 15200
	 
	 setRelevantSphere<<< grid, threads >>>( (float4*)d_cen, d_pos, d_thread, d_Sorted, size);
     //projectSub<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_matrix, d_lamda, size, interval);
     
     cudppDestroyPlan(scanPlan);
     return relevantNum;
}

void ProjectOnAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
	 dim3 threads(512, 1);
     dim3 grid(size/512, 1, 1);
     
	 axisVec<<<1, 1>>>(g_idata1, g_idata2, d_lamda);
     project<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_Sorted, size);
}

void ElongOnAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
	 dim3 threads(512, 1);
     dim3 grid(size/512, 1, 1);
     
	 axisVec<<<1, 1>>>(g_idata1, g_idata2, d_lamda);
     projectSub<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_Sorted, d_matrix, d_lamda, size, interval);
}

void ElongOnAxisMultiple(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval)
{
	 dim3 threads(512, 1);
     dim3 grid(size/512, 1, 1);
     
	 axisVecMultiple<<<1, 1>>>(g_idata1, g_idata2, d_lamda);
     projectSubMultiple<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_Sorted, d_matrix, d_lamda, size, interval);
	 //projectSubMultiple9<<< grid, threads >>>(g_idata1, g_idata2, (float4*)d_cen, d_pos, d_Sorted, d_matrix, d_lamda, size, interval);
}

uint PairsRecover(uint* d_pairs, uint* d_mulpairs, uint* id_thread, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum, uint asig_num)
{
	 uint pairsNum = pairsNum1+pairsNum2;
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 recover<<<grid, threads>>>(d_pairs, d_mulpairs, pairsNum);

	 CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;
	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, pairsNum, 1, 0);
	 cudppScan(scanPlan, d_mulpairs, d_mulpairs, pairsNum);

	 uint totalPairs1 = 0;
	 uint totalPairs = 0;
	 CUDA_SAFE_CALL( cudaMemcpy(&totalPairs1, &(d_mulpairs[pairsNum1-1]), sizeof(udword), cudaMemcpyDeviceToHost) );
	 CUDA_SAFE_CALL( cudaMemcpy(&totalPairs, &(d_mulpairs[pairsNum-1]), sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("totalPairs1 %d\n", totalPairs1);
	 //printf("totalPairs %d\n", totalPairs);
	 setPairsIndex<<<grid, threads>>>(d_mulpairs, id_thread, pairsNum);
	 cutilSafeCall(cudaMemset(d_mulpairs, 0, pairsNum*sizeof(uint)));

	 uint line = overlapNum/2 * objNum;
	 dim3 grid1(ceil(totalPairs1/512.0), 1, 1);
	 pairsMatrix1<<<grid1, threads>>>(d_mulpairs, d_pairs, id_thread, totalPairs1, line, objNum, pairsNum, asig_num);
	 uint totalPairs2 = totalPairs - totalPairs1;
	 dim3 grid2(ceil(totalPairs2/512.0), 1, 1);
	 pairsMatrix2<<<grid2, threads>>>(d_mulpairs, d_pairs, id_thread, totalPairs2, totalPairs1, line, objNum, pairsNum, asig_num);

	 cudppPlan(&scanPlan, config, objNum, 1, 0);
	 cudppScan(scanPlan, d_mulpairs, d_mulpairs, objNum);
	 uint dPairsNum = 0;
	 CUDA_SAFE_CALL( cudaMemcpy(&dPairsNum, &(d_mulpairs[objNum-1]), sizeof(udword), cudaMemcpyDeviceToHost) );
	 //printf("PairsNum %d\n", dPairsNum);

	 dim3 grid3(ceil(objNum/512.0), 1, 1);
	 outputPairs<<<grid3, threads>>>(d_mulpairs, d_pairs, objNum, dPairsNum);
	 cudppDestroyPlan(scanPlan);
	 return dPairsNum;
	 //need reset d_mulpairs to 0
}

void MergePairs(uint* d_pairs, uint* d_mulpairs, uint* d_thread, uint realpairsNum, uint objNum, uint asig_num)
{
	 dim3 threads(512, 1);
     dim3 grid(ceil(realpairsNum/512.0), 1, 1);
	 cutilSafeCall(cudaMemset(d_thread, 0xffffffff, (objNum+1)*sizeof(uint)));
	 reverseIndex<<<grid, threads>>>(d_pairs, d_thread, realpairsNum);
	 //dim3 grid1(ceil(objNum/512.0), 1, 1);
	 //addPairs<<<grid1, threads>>>(d_thread, d_pairs, d_mulpairs, objNum, realpairsNum, asig_num);
}

void replicate(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum)
{
	 uint pairsNum = pairsNum1+pairsNum2;
	 uint line = overlapNum/2 * objNum;
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 duplicate<<<grid, threads>>>(d_pairs, d_mulpairs, pairsNum, line);
}

void objToPairs(uint* d_mulpairs, uint* d_thread, uint pairsNum, uint objNum)
{
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 cutilSafeCall(cudaMemset(d_thread, 0xffffffff, (objNum+1)*sizeof(uint)));
	 reverseIndex<<<grid, threads>>>(d_mulpairs, d_thread, pairsNum);
}

void orderPairs(uint* Sorted, uint* data, uint* sortData, uint pairsNum)
{
	 dim3 threads(512, 1);
	 dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 //reorderPairs<<<grid, threads>>>(g_idata1, g_idata2, g_idata3, pairsNum);
	 reorderData<<<grid, threads>>>(Sorted, data, sortData, pairsNum);
}

void setSortIndex(uint* Sorted, uint numParticles)
{
	dim3 threads(512, 1);
    dim3 grid(ceil(numParticles/512.0), 1, 1);
	setSortKernel<<<grid, threads>>>(Sorted, numParticles);
}

void scaleData(float* data, uint num)
{
	dim3 threads(512, 1);
    dim3 grid(ceil(num/512.0), 1, 1);
	scaleKernel<<<grid, threads>>>((float4*)data, num);
}

void copyPairs(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum)
{
	 uint pairsNum = pairsNum1+pairsNum2;
	 uint line = overlapNum/2 * objNum;
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 duplicateOne<<<grid, threads>>>(d_pairs, d_mulpairs, pairsNum, line);
}

void movePairs(uint* d_mulpairs, uint pairsNum, uint stride1, uint stride2)
{
	 uint size = 2*pairsNum;
	 dim3 threads(512, 1);
     dim3 grid(ceil(size/512.0), 1, 1);
	 duplicateTwo<<<grid, threads>>>(d_mulpairs, size, stride1, stride2);
}

void
cullRepetitivePairs(uint* d_mulpairs, uint* d_pairs, uint* d_pairEntries, uint* d_thread, uint pairsNum, uint* numPairs, uint* numPairEntries)
{
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 cutilSafeCall(cudaMemset(d_pairs, 0, pairsNum*sizeof(uint)));
	 cullIndex<<<grid, threads>>>(d_mulpairs, d_pairs, pairsNum);

	 CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;
	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, pairsNum, 1, 0);
	 cudppScan(scanPlan, d_pairs, d_pairs, pairsNum);
	 uint totalPairs = 0;
	 CUDA_SAFE_CALL( cudaMemcpy(&totalPairs, &(d_pairs[pairsNum-1]), sizeof(uint), cudaMemcpyDeviceToHost) );
	 //printf("totalPairs:: %-5d\n", totalPairs);
	 CUDA_SAFE_CALL( cudaMemcpy(numPairs, &(d_pairs[pairsNum-1]), sizeof(uint), cudaMemcpyDeviceToDevice) );
	 CUDA_SAFE_CALL( cudaMemcpy(d_mulpairs+2*pairsNum, d_pairs, sizeof(uint)*pairsNum, cudaMemcpyDeviceToDevice) );
	 dim3 grid1(ceil(totalPairs/512.0), 1, 1);
	 setCullPairs<<<grid, threads>>>(d_mulpairs, d_pairs, pairsNum, totalPairs);

	 cutilSafeCall(cudaMemset(d_mulpairs, 0, totalPairs*sizeof(uint)));
	 setPairsEntriesFlag<<<grid1, threads>>>(d_pairs, d_mulpairs, totalPairs);
     uint nPairEntries = 0;
	 cudppPlan(&scanPlan, config, totalPairs, 1, 0);
	 cudppScan(scanPlan, d_mulpairs, d_mulpairs, totalPairs);
	 //CUDA_SAFE_CALL( cudaMemcpy(&nPairEntries, &(d_mulpairs[totalPairs-1]), sizeof(uint), cudaMemcpyDeviceToHost) );
	 CUDA_SAFE_CALL( cudaMemcpy(numPairEntries, &(d_mulpairs[totalPairs-1]), sizeof(uint), cudaMemcpyDeviceToDevice) );
	 setPairsEntries<<<grid1, threads>>>(d_mulpairs, d_pairs, d_pairEntries, totalPairs);

	 //*numPairEntries = nPairEntries;
	 //*numPairs =  totalPairs;
	 cudppDestroyPlan(scanPlan);
}

uint
extrPairsEntries(uint* d_mulpairs, uint* d_pairs, uint* d_pairEntries, uint* d_thread, uint pairsNum)
{
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
     //CUDA_SAFE_CALL( cudaMemcpy(d_pairs, d_mulpairs, sizeof(uint)*2*pairsNum, cudaMemcpyDeviceToDevice) );
	 setCullPairsDirect<<<grid, threads>>>(d_mulpairs, d_pairs, pairsNum);

	 cutilSafeCall(cudaMemset(d_mulpairs, 0, pairsNum*sizeof(uint)));
	 setPairsEntriesFlag<<<grid, threads>>>(d_pairs, d_mulpairs, pairsNum);
     uint nPairEntries = 0;
	 CUDPPConfiguration config;
	 config.algorithm = CUDPP_SCAN;
	 config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	 config.op = CUDPP_ADD;
	 config.datatype = CUDPP_UINT;
	 CUDPPHandle scanPlan;
	 cudppPlan(&scanPlan, config, pairsNum, 1, 0);
	 cudppScan(scanPlan, d_mulpairs, d_mulpairs, pairsNum);
	 CUDA_SAFE_CALL( cudaMemcpy(&nPairEntries, &(d_mulpairs[pairsNum-1]), sizeof(uint), cudaMemcpyDeviceToHost) );
	 //CUDA_SAFE_CALL( cudaMemcpy(numPairEntries, &(d_mulpairs[pairsNum-1]), sizeof(uint), cudaMemcpyDeviceToDevice) );
	 setPairsEntries<<<grid, threads>>>(d_mulpairs, d_pairs, d_pairEntries, pairsNum);

	 //*numPairEntries = nPairEntries;
	 //*numPairs =  totalPairs;
	 cudppDestroyPlan(scanPlan);
	 return nPairEntries;
}

void copyPairsWithoutSub(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum)
{
	 uint pairsNum = pairsNum1+pairsNum2;
	 uint line = overlapNum/2 * objNum;
	 dim3 threads(512, 1);
     dim3 grid(ceil(pairsNum/512.0), 1, 1);
	 duplicateInterleave<<<grid, threads>>>(d_pairs, d_mulpairs, pairsNum, line);
}

}   // extern "C"
