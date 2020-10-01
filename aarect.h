#pragma once

#include "hitable.h"
#include "material.h"

class xyrect : public hitable {
	public:
		__device__ xyrect() {}
		
		__device__ xyrect(float _x1, float _x2, float _y1, float _y2,
			float _k, material* mat)
			: x0(_x1), x1(_x2), y0(_y1), y1(_y2), k(_k), mat_ptr(mat) {};
        
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;

	public:
		float x0, x1, y0, y1, k;
		material* mat_ptr;
};

__device__ bool xyrect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}
