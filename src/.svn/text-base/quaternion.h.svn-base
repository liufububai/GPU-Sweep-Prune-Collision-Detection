/**
*	@author Takahiro HARADA
*/
#ifndef _QUATERNION_
#define _QUATERNION_

#include <vector_types.h>
#include <vector_functions.h>

#include "../include/matrix.h"


//==== VECTOR ====
inline
__device__ __host__
float4 cross(float4 a, float4 b){
	float4 ans;
	ans.x=a.y*b.z-a.z*b.y;
	ans.y=a.z*b.x-a.x*b.z;
	ans.z=a.x*b.y-a.y*b.x;
	ans.w=0;
	return ans;
}

inline
#ifndef CUTIL_MATH_H
__device__ __host__
float dot(float4 a, float4 b){
	return a.x*b.x+a.y*b.y+a.z*b.z;
}
inline
__device__ __host__
float length(float4 a){
	return sqrt(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}
#endif

//==== QUATERNION ====
inline
__device__ __host__
float4 mulQuaternion(float4 a,float4 b){
	float4 ans;
	ans=cross(a,b);
	ans.x+=a.w*b.x+b.w*a.x;
	ans.y+=a.w*b.y+b.w*a.y;
	ans.z+=a.w*b.z+b.w*a.z;

	ans.w=a.w*b.w-dot(a,b);
	return ans;
}

inline
__device__ __host__
float4 invertQuaternion(float4 a){
	return make_float4(-a.x,-a.y,-a.z,a.w);
}
inline
__device__ __host__
float4 normalizeQuaternion(float4 a){
	float qLength=length(a);

	return make_float4(a.x/qLength,a.y/qLength,a.z/qLength,a.w/qLength);
}
inline
__device__ __host__
float4 rotateVectorWQuaternion(float4 quat,float4 inVec){
	inVec.w=0;
	float4 quatInv = invertQuaternion(quat);
	float4 rotatedPos = mulQuaternion(mulQuaternion(quat,inVec),quatInv);
	return rotatedPos;
}
inline
__device__ __host__
float4 w2Quaternion(float4 w){
	float4 ans;
	w.w=0;
	float lw=length(w);

	ans.x=w.x/lw;
	ans.y=w.y/lw;
	ans.z=w.z/lw;

	ans.w = lw*0.5f;
	ans.x *= sin(ans.w);
	ans.y *= sin(ans.w);
	ans.z *= sin(ans.w);
	ans.w = cos(ans.w);

	if(lw < 0.0001)
		ans=make_float4(0,0,0,1);

	return ans;
}

inline
__device__ __host__
Matrix quat2Matrix(float4 quat){
	Matrix matrix;

	float3 quat2=make_float3(quat.x*quat.x,quat.y*quat.y,quat.z*quat.z);
	matrix.e[0][0]=1-2*quat2.y-2*quat2.z;
	matrix.e[0][1]=2*quat.x*quat.y-2*quat.w*quat.z;
	matrix.e[0][2]=2*quat.x*quat.z+2*quat.w*quat.y;
	matrix.e[1][0]=2*quat.x*quat.y+2*quat.w*quat.z;
	matrix.e[1][1]=1-2*quat2.x-2*quat2.z;
	matrix.e[1][2]=2*quat.y*quat.z-2*quat.w*quat.x;
	matrix.e[2][0]=2*quat.x*quat.z-2*quat.w*quat.y;
	matrix.e[2][1]=2*quat.y*quat.z+2*quat.w*quat.x;
	matrix.e[2][2]=1-2*quat2.x-2*quat2.y;

	return matrix;
}

inline
__device__ __host__
Matrix transpose(Matrix m){
	Matrix a;
	
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			a.e[i][j]=m.e[j][i];

	return a;
}

inline
__device__ __host__
Matrix mul(Matrix a, Matrix b){
	Matrix ans;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			ans.e[i][j]=0;

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				ans.e[i][j]+=a.e[i][k]*b.e[k][j];
//				ans.e[j][k]+=a.e[i][k]*b.e[j][i];

	return ans;
}
inline
__device__ __host__
float4 mulMatrixVector(Matrix a, float4 b){
	return make_float4(
		a.e[0][0]*b.x+a.e[1][0]*b.y+a.e[2][0]*b.z,
		a.e[0][1]*b.x+a.e[1][1]*b.y+a.e[2][1]*b.z,
		a.e[0][2]*b.x+a.e[1][2]*b.y+a.e[2][2]*b.z,
		0);
}


/*
//========================
//	DEVICE
//========================

//==== VECTOR ====
__device__ float4 crossD(float4 a, float4 b){
	float4 ans;
	ans.x=a.y*b.z-a.z*b.y;
	ans.y=a.z*b.x-a.x*b.z;
	ans.z=a.x*b.y-a.y*b.x;
	ans.w=0;
	return ans;
}
__device__ float dotD(float4 a, float4 b){
	return a.x*b.x+a.y*b.y+a.z*b.z;
}
__device__ float lengthD(float4 a){
	return sqrt(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}
//==== QUATERNION ====
__device__ float4 mulQuaternionD(float4 a,float4 b){
	float4 ans;
	ans=crossD(a,b);
	ans.x+=a.w*b.x+b.w*a.x;
	ans.y+=a.w*b.y+b.w*a.y;
	ans.z+=a.w*b.z+b.w*a.z;

	ans.w=a.w*b.w-dotD(a,b);
	return ans;
}
__device__ float4 invertQuaternionD(float4 a){
	return make_float4(-a.x,-a.y,-a.z,a.w);
}
__device__ float4 normalizeQuaternionD(float4 a){
	float qLength=lengthD(a);

	return make_float4(a.x/qLength,a.y/qLength,a.z/qLength,a.w/qLength);
}
__device__ float4 rotateVectorWQuaternionD(float4 quat,float4 inVec){
	inVec.w=0;
	float4 quatInv = invertQuaternionD(quat);
	float4 rotatedPos = mulQuaternionD(mulQuaternionD(quat,inVec),quatInv);
	return rotatedPos;
}
__device__ float4 w2QuaternionD(float4 w){
	float4 ans;
	w.w=0;
	float lw=lengthD(w);

	ans.x=w.x/lw;
	ans.y=w.y/lw;
	ans.z=w.z/lw;

	ans.w = lw*0.5f;
	ans.x *= sin(ans.w);
	ans.y *= sin(ans.w);
	ans.z *= sin(ans.w);
	ans.w = cos(ans.w);

	if(lw < 0.0001)
		ans=make_float4(0,0,0,1);

	return ans;
}

__device__ Matrix quat2MatrixD(float4 quat){
	Matrix matrix;

	float3 quat2=make_float3(quat.x*quat.x,quat.y*quat.y,quat.z*quat.z);
	matrix.e[0][0]=1-2*quat2.y-2*quat2.z;
	matrix.e[0][1]=2*quat.x*quat.y-2*quat.w*quat.z;
	matrix.e[0][2]=2*quat.x*quat.z+2*quat.w*quat.y;
	matrix.e[1][0]=2*quat.x*quat.y+2*quat.w*quat.z;
	matrix.e[1][1]=1-2*quat2.x-2*quat2.z;
	matrix.e[1][2]=2*quat.y*quat.z-2*quat.w*quat.x;
	matrix.e[2][0]=2*quat.x*quat.z-2*quat.w*quat.y;
	matrix.e[2][1]=2*quat.y*quat.z+2*quat.w*quat.x;
	matrix.e[2][2]=1-2*quat2.x-2*quat2.y;

	return matrix;
}
__device__ Matrix transposeD(Matrix m){
	Matrix a;
	
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			a.e[i][j]=m.e[j][i];

	return a;
}
__device__ Matrix mulD(Matrix a, Matrix b){
	Matrix ans;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			ans.e[i][j]=0;

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				ans.e[i][j]+=a.e[i][k]*b.e[k][j];
//				ans.e[j][k]+=a.e[i][k]*b.e[j][i];

	return ans;
}
__device__ float4 mulMatrixVectorD(Matrix a, float4 b){
	return make_float4(
		a.e[0][0]*b.x+a.e[1][0]*b.y+a.e[2][0]*b.z,
		a.e[0][1]*b.x+a.e[1][1]*b.y+a.e[2][1]*b.z,
		a.e[0][2]*b.x+a.e[1][2]*b.y+a.e[2][2]*b.z,
		0);
}
*/

#endif
