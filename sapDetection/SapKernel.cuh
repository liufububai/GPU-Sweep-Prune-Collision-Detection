typedef unsigned int uint;
typedef unsigned int		udword;	

extern "C"
{
uint sweepPrunSphereSubCross(float* d_cen, float* d_pos,  uint* d_Sorted,  uint* d_pairs, uint* d_thread, uint* id_thread, 
                                         uint* d_mulpairs, uint nbox, uint rbox, uint offset, uint overlapnum, uint asig_num, 
										 uint avthreadNum, float* sortedCen, int* d_hash, float* d_mean, int* d_lamda, float interval, float* d_matrix);

uint sweepPrunSphereSubElong(float* d_cen, float* d_pos,  uint* d_Sorted,  uint* d_pairs, uint* d_thread, uint* id_thread, 
                                          uint* d_mulpairs, uint nbox, uint rbox, uint overlapnum, uint asig_num, uint avthreadNum, 
										  float* sortedCen, int* d_hash, float* d_mean, int* d_lamda, float interval, float* d_matrix);

void reduce(int size, int threads, int blocks, int whichKernel, float* d_idata, float* d_odata);

void calDelta(float *d_mean, float *g_idata, float *g_odata1, float *g_odata2, int size);

void projectData(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, float* d_matrix, int* d_lamda, int size);

uint optimalAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, uint* d_thread, int* d_num, float* d_radii, int size, float interval);

uint optimalAxisMultiple(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, uint* d_thread, int* d_num, float* d_radii, int size, float interval);

void ProjectOnAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, udword* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval);

void ElongOnAxis(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval);

void ElongOnAxisMultiple(float *g_idata1, float *g_idata2, float *d_cen, float *d_pos, uint* d_Sorted, float* d_matrix, int* d_lamda, int size, float interval);

uint PairsRecover(uint* d_pairs, uint* d_mulpairs, uint* id_thread, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum, uint asig_num);

void replicate(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum);

void orderPairs(uint* g_idata1, uint* g_idata2, uint* g_idata3, uint pairsNum);

void MergePairs(uint* d_pairs, uint* d_mulpairs, uint* d_thread, uint realpairsNum, uint objNum, uint asig_num);

void objToPairs(uint* d_mulpairs, uint* d_thread, uint pairsNum, uint objNum);

void updatePosVel(float *pos, float *vel, float *newVel, uint* d_thread, uint* d_mulpairs, uint numParticles, uint pairsNum);

void setSortIndex(uint* d_Sorted, uint numParticles);

void scaleData(float* data, uint num);

void copyPairs(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum);

void copyPairsWithoutSub(uint* d_pairs, uint* d_mulpairs, uint pairsNum1, uint pairsNum2, uint objNum, uint overlapNum);

void movePairs(uint* d_mulpairs, uint pairsNum, uint stride1, uint stride2);

void cullRepetitivePairs(uint* d_mulpairs, uint* d_pairs, uint* d_pairEntries, uint* d_thread, uint pairsNum, uint* numPairs, uint* numPairEntries);

uint extrPairsEntries(uint* d_mulpairs, uint* d_pairs, uint* d_pairEntries, uint* d_thread, uint pairsNum);

}