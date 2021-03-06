/**
*
*	@author Takahiro HARADA
*
*/
#include "ParticleRigidBodyWrapper.h"
#include "ParticleRigidBodyKernel.h"


void PRBWrap::integrate(dim3 dimGrid,dim3 dimBlock, 
			float4* pos, Quaternion* quat, 
			float4* linVel, float4* angVel, 
			float dt, int size)
{
	integrateK<<<dimGrid, dimBlock>>>(pos, quat, linVel, angVel, dt, size);
}

void PRBWrap::upadteParticleData(dim3 dimGrid,dim3 dimBlock, 
			float4* pos, Quaternion* quat, 
			float4* linVel, float4* angVel, 
			int* shapeIdxBuf, int* pBufStartOffsetBuf, int* numParticles, 
			float4* particleRefPosBuf, float* particleRefMassBuf,
			float4* particlePosOut, float4* particleVelOut, float* particleMassOut, int size)
{
	updateParticleDataK<<<dimGrid, dimBlock>>>(pos, quat, linVel, angVel, 
		shapeIdxBuf, pBufStartOffsetBuf, numParticles, particleRefPosBuf, particleRefMassBuf,
		particlePosOut, particleVelOut, particleMassOut, size);
}

void PRBWrap::clearForceAndBoundaryCondition(dim3 dimGrid,dim3 dimBlock, 
			float4* pPos, float4* pVel, float* pMass, 
			float4* pForceOut, 
			float spMultip, float dpMultip, int numParticles, float dt)
{
	clearForceAndBoundaryConditionK<<<dimGrid, dimBlock>>>( pPos, pVel, pMass, pForceOut, 
		spMultip, dpMultip, numParticles, dt );
}

void PRBWrap::updateVelocity(dim3 dimGrid,dim3 dimBlock, 
			float4* linVel, float4* angVel, 
			Quaternion* quat, float* invMass, Matrix3x3* invInertia, 
			int* shapeIdxBuf, int* pBufStartOffsetBuf, int* numParticles, 
			float4* particleRefPosBuf, 
			float4* particleForceBuf, int numBodies, float dt,
			float maxLinVel, bool capVel)
{
	if( capVel )
	{
		updateVelocityK<true><<<dimGrid, dimBlock>>>(linVel, angVel, quat, invMass, invInertia, shapeIdxBuf, pBufStartOffsetBuf, numParticles,
			particleRefPosBuf, particleForceBuf, numBodies, dt, maxLinVel);
	}
	else
	{
		updateVelocityK<false><<<dimGrid, dimBlock>>>(linVel, angVel, quat, invMass, invInertia, shapeIdxBuf, pBufStartOffsetBuf, numParticles,
			particleRefPosBuf, particleForceBuf, numBodies, dt, maxLinVel);
	}
}

void PRBWrap::prepare(dim3 dimGrid,dim3 dimBlock, 
			Quaternion* quat, int* shapeIdxBuf,
			Matrix3x3* shapeInvInertia, 
			Matrix3x3* rbInvInertiaOut, int size)
{
	prepareK<<<dimGrid, dimBlock>>>(quat, shapeIdxBuf, shapeInvInertia, rbInvInertiaOut, size);
}

void PRBWrap::computeFoceOnParticles(dim3 dimGrid,dim3 dimBlock, 
			float4* pPos, float4* pVel, 
			int2* pairs, int* pairEntries, int numPairEntries, 
			float* particleMass,
			float spMultip, float dpMultip, 
			float4* pForceOut, float dt)
{
	comupteFoceOnParticlesK<<<dimGrid, dimBlock>>>(pPos, pVel, pairs, pairEntries, numPairEntries, particleMass, 
		spMultip, dpMultip, 
		pForceOut, dt);
}

