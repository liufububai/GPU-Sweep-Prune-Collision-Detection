/**
*
*	@author Takahiro HARADA
*
*/

#ifndef RIGID_BODY_SYSTEM_H
#define RIGID_BODY_SYSTEM_H

#ifdef _WIN32
#ifdef RIGIDBODYSYSTEM_EXPORTS
#define DLL_API_RIGIDBODYSYSTEM __declspec(dllexport)
#else
#define DLL_API_RIGIDBODYSYSTEM __declspec(dllimport)
#endif
#pragma comment(lib,"C:/CUDA/lib/cudart.lib")
#else
#define DLL_API_RIGIDBODYSYSTEM
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <cutil.h>

#include "mathConstants.h"
#include "Matrix3x3.h"

#define MAX_SAMPLES  24
#define DF_RESOLUTION  32
#define MAX_CONTACTS 4
#define MAX_SHAPES 5
#define NUM_CONSTRAINTS_PER_BLOCK 256
#define SOLVE_GROUP_SIZE 64
#define MAX_BATCH_SIZE 6
#define COLORING_BUFFER_SIZE 64
#define COLORING_BUFFER_MASK (COLORING_BUFFER_SIZE-1)


typedef unsigned char uint8;

class Aabb;
struct Quaternion;

class 
DLL_API_RIGIDBODYSYSTEM
VecUtils
{
	public:
		static void rotateVector(const float4& quat, const float4& in, float4& out);
};

struct 
DLL_API_RIGIDBODYSYSTEM
__builtin_align__(16)
ShapeData
{
	static const float s_infMass;

	float m_dfScaleInv;
	float m_dx;
	float m_invMass;
	float m_padding0;
	Matrix3x3 m_invInertia;

	inline
	__device__ __host__
	float4 transformToGridCrd(float4& p) const;

	inline
	__device__ __host__
	static float4 transformToGridCrd(float dfScaleInv, float4& p);
};

//	world space
struct
DLL_API_RIGIDBODYSYSTEM
__builtin_align__(16)
Contact
{
	float4 m_pos;
	float4 m_normal;
	float m_penetration;
	float m_lambdaDt;
	float m_padding0;
	float m_padding1;
};

class
DLL_API_RIGIDBODYSYSTEM
RigidBodySystem
{
	public:
		struct
		DLL_API_RIGIDBODYSYSTEM
		BodyData
		{
			int m_maxSize;
			int m_size;
			float4* m_pos;
			Quaternion* m_quat;
			float4* m_linVel;
			float4* m_angVel;
			int* m_shapeIdx;
			ShapeData* m_shapeData;
			Matrix3x3* m_invInertia;

			Aabb* m_aabbs;

			float4* m_samplePoints;

			BodyData() : m_size(0){}

			void memAllocCpu(int maxSize);
			void memFreeCpu();

			void memAllocGpu(int maxSize);
			void memFreeGpu();
		};

		struct Jacobian
		{
			float4 m_linear[MAX_CONTACTS];
			float4 m_angular0[MAX_CONTACTS];
			float4 m_angular1[MAX_CONTACTS];
			float m_jacCoeffInv[MAX_CONTACTS];
			float m_b[MAX_CONTACTS];
			float m_appliedRambdaDt[MAX_CONTACTS];
		};

		struct JacobianArray
		{
			float4* m_linear;
			float4* m_angular0;
			float4* m_angular1;
			float* m_jacCoeffInv;
			float* m_b;
			float* m_appliedRambdaDt;
			int m_maxSize;

			void memAllocCpu(int maxSize);
			void memFreeCpu();

			void memAllocGpu(int maxSize);
			void memFreeGpu();
		};

		struct
		DLL_API_RIGIDBODYSYSTEM
		SimConfig
		{
			SimConfig() : m_nSolverIter(4) {}
			int m_nSolverIter;
		};

		enum ProcType
		{
			SIM_CPU,
			SIM_GPU,
		};

		RigidBodySystem(int maxNumBodies, int maxNumPairs, ProcType procType = SIM_GPU) ;
		~RigidBodySystem();

		enum AddSampleType
		{
			SAMPLE_NON, 
			SAMPLE_FACE_CENTER,
		};

		//	interface for broadphase
		const Aabb* getAabbs() { return m_bodyData.m_aabbs; }
		float4* getBoundingSphere() { return m_bodyData.m_pos; }//m_boundingSphere; }
		int2* getBroadphasePairPtr(){ return m_pairs; }
		void setNumPairs(int numPairs) { m_numPairs = numPairs; }
		int getNumPairs() const { return m_numPairs; }

		//	<< procedure >>
		//	calcAabb()
		//	broadphase()
		//	step()

		//	after setting all the shapes, call compile before simulation start
		void compile();

		void calcAabb();
		void calcBoundingSphere();
		void step(float dt);
		void prepare();
		void narrowPhase();
		void solve(float dt);
		void integrate(float dt);
		void updateSamples();

		//	interface for shapes
		void setShape(int shapeIdx, 
			int3* triIdx, float4* vtx, int nTri, int nVtx, float density=1.f, AddSampleType sampleType = SAMPLE_NON);





		inline
		__device__ __host__
		static int getCrd(int shapeIdx, int x, int y, int z)
		{
			int base = shapeIdx*DF_RESOLUTION*DF_RESOLUTION*DF_RESOLUTION;
			return base + x+y*DF_RESOLUTION+z*DF_RESOLUTION*DF_RESOLUTION; 
		}

		inline
		__device__ __host__
		static float dfSample(int shapeIdx, float dx, float* shapeDfBuffer,
			float4 x, float4& n);

		inline
		__device__ __host__
		static int getSampleIdx(int shapeIdx, int iVtx)  { return shapeIdx*MAX_SAMPLES+iVtx; }


		//	setter / getter
		void set(float4* pos, float4* quat, float4* linVel, float4* angVel, 
			int* shapeIdx, int nBodies);
		void setShapeData(ShapeData* shapeData, int nShapeData);
		int getNumBodies() const { return m_bodyData.m_size; }
		void cpyBodyData(float4* pos, float4* quat, int nBodies) const;
		void cpyMaxMin(float4* aabbMax, float4* aabbMin, int nBodies) const;
		void cpyNumContacts(int* num, int numData) const;
		SimConfig& getSimConfig() { return m_simConfig; }

		//	only for debugging
		void showContacts() const;
		void showSamples() const;
		void showBatches() const;

	public:
		BodyData m_bodyData;
		JacobianArray m_jacArray;
		uint8* m_batchIdx;
		int2* m_pairs;
		int m_numPairs;
		Contact* m_contacts;
		uint8* m_numContacts;
		float4* m_boundingSphere;

		float* m_shapeDfBuffer;
		float4* m_shapeSampleBuffer;

		float* m_shapeDfBufferCpu;
		float4* m_shapeSampleBufferCpu;
		ShapeData* m_shapeDataCpu; //

		ProcType m_procType;
		SimConfig m_simConfig;
};

inline
__device__ __host__
float4 ShapeData::transformToGridCrd(float4& p) const
{
	float4 v = (p*m_dfScaleInv + make_float4(0.5f))*DF_RESOLUTION;
	v.w = 0.f;
	return v;
}

inline
__device__ __host__
float4 ShapeData::transformToGridCrd(float dfScaleInv, float4& p)
{
	float4 v = (p*dfScaleInv + make_float4(0.5f))*DF_RESOLUTION;
	v.w = 0.f;
	return v;
}

/*
inline
__device__ __host__
float ShapeData::sample(float4 x, float4& n) const
{
	x -= make_float4(0.5f);
	if( x.x <= 0 || x.y <= 0 || x.z <= 0 ||
		x.x >= DF_RESOLUTION-1 || x.y >= DF_RESOLUTION-1 || x.z >= DF_RESOLUTION-1 ) return 2;
	int4 crd;
	crd.x = int(x.x); crd.y = int(x.y); crd.z = int(x.z);
	float4 frac;
	frac.x = x.x-crd.x; frac.y = x.y-crd.y; frac.z = x.z-crd.z;

	float s[8];
	s[0] = m_df[getCrd(crd.x, crd.y, crd.z)];
	s[1] = m_df[getCrd(crd.x+1, crd.y, crd.z)];
	s[2] = m_df[getCrd(crd.x, crd.y+1, crd.z)];
	s[3] = m_df[getCrd(crd.x+1, crd.y+1, crd.z)];

	s[4] = m_df[getCrd(crd.x, crd.y, crd.z+1)];
	s[5] = m_df[getCrd(crd.x+1, crd.y, crd.z+1)];
	s[6] = m_df[getCrd(crd.x, crd.y+1, crd.z+1)];
	s[7] = m_df[getCrd(crd.x+1, crd.y+1, crd.z+1)];

	float ansM = (s[0]*(1.f-frac.x)+s[1]*frac.x)*(1.f-frac.y) + (s[2]*(1.f-frac.x)+s[3]*frac.x);
	float ansP = (s[4]*(1.f-frac.x)+s[5]*frac.x)*(1.f-frac.y) + (s[6]*(1.f-frac.x)+s[7]*frac.x);
	float ans = ansM*(1.f-frac.z) + ansP*frac.z;

	n.x = -s[1]+s[0];
	n.y = -s[2]+s[0];
	n.z = -s[4]+s[0];
	n.w = 0.f;
	if( n.x*n.x+n.y*n.y+n.z*n.z < EPSILON ) return 2;
	n = normalize(n);
	return ans*m_dx;
}
*/
inline
__device__ __host__
float RigidBodySystem::dfSample(int shapeIdx, float dx, float* shapeDfBuffer,
								float4 x, float4& n) 
{
	const float epsilon = 0.0001f;

	x -= make_float4(0.5f);
	bool invalid = ( x.x <= 0 || x.y <= 0 || x.z <= 0 ||
		x.x >= DF_RESOLUTION-1 || x.y >= DF_RESOLUTION-1 || x.z >= DF_RESOLUTION-1 );

//	if( x.x <= 0 || x.y <= 0 || x.z <= 0 ||
//		x.x >= DF_RESOLUTION-1 || x.y >= DF_RESOLUTION-1 || x.z >= DF_RESOLUTION-1 ) return 2;

	int4 crd; crd.x = int(x.x); crd.y = int(x.y); crd.z = int(x.z);
	float4 frac; frac.x = x.x-crd.x; frac.y = x.y-crd.y; frac.z = x.z-crd.z;
	float ans;

	float s[8];
	s[0] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x, crd.y, crd.z)];
	s[1] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x+1, crd.y, crd.z)];
	s[2] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x, crd.y+1, crd.z)];
	s[3] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x+1, crd.y+1, crd.z)];

	s[4] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x, crd.y, crd.z+1)];
	s[5] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x+1, crd.y, crd.z+1)];
	s[6] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x, crd.y+1, crd.z+1)];
	s[7] = (invalid)? 0.f: shapeDfBuffer[getCrd(shapeIdx, crd.x+1, crd.y+1, crd.z+1)];

	float ansM = (s[0]*(1.f-frac.x)+s[1]*frac.x)*(1.f-frac.y) + (s[2]*(1.f-frac.x)+s[3]*frac.x);
	float ansP = (s[4]*(1.f-frac.x)+s[5]*frac.x)*(1.f-frac.y) + (s[6]*(1.f-frac.x)+s[7]*frac.x);
	ans = ansM*(1.f-frac.z) + ansP*frac.z;

	n.x = -s[1]+s[0];
	n.y = -s[2]+s[0];
	n.z = -s[4]+s[0];
	n.w = 0.f;

	if( n.x*n.x+n.y*n.y+n.z*n.z < epsilon ) invalid = true;
	n = normalize(n);
//	float dx = shapeData[shapeIdx].m_dx;
	return (invalid)? 2.f : ans*dx;
}

#endif

