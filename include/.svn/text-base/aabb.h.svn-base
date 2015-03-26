/**
*
*	@author Takahiro HARADA
*
*/
#ifndef AABB_H
#define AABB_H

class Aabb
{
	public:
		inline
		void includePoint(float4 p);

		inline
		bool overlaps(const Aabb& v) const;

	public:
		float4 m_max;
		float4 m_min;

};

inline 
float maxf(float a, float b)
{
  return a > b ? a : b;
}

inline 
float minf(float a, float b)
{
  return a < b ? a : b;
}

void Aabb::includePoint(float4 p)
{
	m_max.x = maxf(p.x, m_max.x);
	m_max.y = maxf(p.y, m_max.y);
	m_max.z = maxf(p.z, m_max.z);

	m_min.x = minf(p.x, m_min.x);
	m_min.y = minf(p.y, m_min.y);
	m_min.z = minf(p.z, m_min.z);
}

bool Aabb::overlaps(const Aabb& aabb) const
{
//	float4 result = (m_min <= aabb.m_max) & (aabb.m_min <= m_max);
//	return result.x != 0 && result.y != 0 && result.z != 0;

	bool x = m_min.x <= aabb.m_max.x && aabb.m_min.x <= m_max.x;
	bool y = m_min.y <= aabb.m_max.y && aabb.m_min.y <= m_max.y;
	bool z = m_min.z <= aabb.m_max.z && aabb.m_min.z <= m_max.z;

	return x&&y&&z;
}

#endif

