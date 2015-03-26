/**
*	@author Takahiro HARADA
*/
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "matrix.h"


Matrix Matrix::operator*(const Matrix in){
	Matrix a;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			a.e[i][j]=0;

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			for(int k=0;k<3;k++)
				//					ans.e[i][j]+=a.e[i][k]*b.e[k][j];
				a.e[i][j]+=e[i][k]*in.e[k][j];

	return a;
}
Matrix Matrix::operator/(const float in1){
	Matrix a;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			a.e[i][j]=e[i][j]/in1;
	return a;	
}
Matrix& Matrix::operator/=(const float in){
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			e[i][j]/=in;
	return *this;
}
Matrix Matrix::transpose(){
	Matrix a;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			a.e[i][j]=e[j][i];

	return a;
}
Matrix Matrix::inverse(){
	Matrix a;
	float det;
	a.e[0][0] = e[1][1] * e[2][2] - e[1][2] * e[2][1];
	a.e[0][1] = e[0][2] * e[2][1] - e[0][1] * e[2][2];
	a.e[0][2] = e[0][1] * e[1][2] - e[0][2] * e[1][1];

	a.e[1][0] = e[1][2] * e[2][0] - e[1][0] * e[2][2];
	a.e[1][1] = e[0][0] * e[2][2] - e[0][2] * e[2][0];
	a.e[1][2] = e[0][2] * e[1][0] - e[0][0] * e[1][2];

	a.e[2][0] = e[1][0] * e[2][1] - e[1][1] * e[2][0];
	a.e[2][1] = e[0][1] * e[2][0] - e[0][0] * e[2][1];
	a.e[2][2] = e[0][0] * e[1][1] - e[0][1] * e[1][0];

	det = a.e[0][0] * e[0][0] + a.e[1][0] * e[0][1] + a.e[2][0] * e[0][2];	
	a/=det;
	return a;
}


#endif

