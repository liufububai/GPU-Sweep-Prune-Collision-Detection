/**
*	@author Takahiro HARADA
*/
#ifndef _COMMON_KERNEL_
#define _COMMON_KERNEL_

#include "../include/gridProperties.h"
#include "cutil_math.h"


inline __device__ int getGridDimXIndex(float4 pos, GridProperties gProp)
{
	return floor((pos.x-gProp.minPos.x)/gProp.sideLength);
}
inline __device__ int getGridDimYIndex(float4 pos, GridProperties gProp)
{
	return floor((pos.y-gProp.minPos.y)/gProp.sideLength);
}
inline __device__ int getGridDimZIndex(float4 pos, GridProperties gProp)
{
	return floor((pos.z-gProp.minPos.z)/gProp.sideLength);
}
inline __device__ int get1DGridIndex(int3 grid3DIndex,GridProperties gProp)
{
	return grid3DIndex.x*gProp.res.y*gProp.res.z
		+grid3DIndex.y*gProp.res.z
		+grid3DIndex.z;
}
__device__ int getGridIndex(float4 pos, GridProperties gProp)
{
	int gIndex;
	int3 gridIndex;

	gridIndex.x=getGridDimXIndex(pos,gProp);
	gridIndex.y=getGridDimYIndex(pos,gProp);
	gridIndex.z=getGridDimZIndex(pos,gProp);

	if(gridIndex.x < 0 || gridIndex.y < 0 || gridIndex.z < 0)
		return -1;
	if(gridIndex.x >= gProp.res.x || gridIndex.y >= gProp.res.y || gridIndex.z >= gProp.res.z)
		return -1;

	gIndex=get1DGridIndex(gridIndex,gProp);

	return gIndex;
}

//	Unroll to optimize
inline __host__ __device__ int getElement(int4 a, int b)
{
	switch (b)
	{
	case 0 :
		return a.x;
	case 1:
		return a.y;
	case 2:
		return a.z;
	case 3:
		return a.w;
	}
	return 0;
}
//	Unroll to optimize
inline __host__ __device__ float getElement(float4 a, int b)
{
	switch (b)
	{
	case 0 :
		return a.x;
	case 1:
		return a.y;
	case 2:
		return a.z;
	case 3:
		return a.w;
	}
	return 0.0f;
}
#endif


