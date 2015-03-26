/**
*
*	@author Takahiro HARADA
*
*/
#ifndef TRANSFER_UTIL_H
#define TRANSFER_UTIL_H

class TransferUtils
{
	public:
		inline
		static void allocateAndCpyH2D(void*& devicePtr, void* hostPtr, size_t size)
		{
			cudaMalloc((void**)&devicePtr, size);
			cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
		}

		inline
		static void cpyAndFreeD2H(void* devicePtr, void* hostPtr, size_t size)
		{
			cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
			cudaFree(devicePtr);
		}

		inline
		static void allocateAndCpyD2H(void*&hostPtr, void* devicePtr, size_t size)
		{
			hostPtr = (void*) malloc(size);
			cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
		}

		inline
		static void cpyAndFreeH2D(void* hostPtr, void* devicePtr, size_t size)
		{
			cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
			free( hostPtr );
		}

		inline
		static void copyHostToDevice(void* devicePtr, void* hostPtr, size_t size)
		{
			cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
		}

		inline
		static void copyDeviceToHost(void* hostPtr, void* devicePtr, size_t size)
		{
			cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
		}

};

#endif
