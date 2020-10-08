#pragma once

#include "hitable.h"
#include "material.h"

class xyrect : public hitable {
	public:
		__device__ xyrect() {}
		
		__device__ xyrect(float _x1, float _x2, float _y1, float _y2,
			float _k, material* mat,bool _flipface=false)
			: x0(_x1), x1(_x2), y0(_y1), y1(_y2), k(_k), mat_ptr(mat), flipface(_flipface) {};
        
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;

	public:
		float x0, x1, y0, y1, k;
        bool flipface;
		material* mat_ptr;
};



class xzrect : public hitable {
public:
    __device__ xzrect() {}
    __device__ xzrect(float _x0, float _x1, float _z0, float _z1, float _k, material* m, bool _flipface = false)
    :x0(_x0),x1(_x1),z0(_z0),z1(_z1),k(_k),mp(m),flipface(_flipface){};

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const override;
public:
    material* mp;
    float x0, x1, z0, z1, k;
    bool flipface;

};

class yzrect : public hitable {
public:
    __device__ yzrect() {}
    __device__ yzrect(float _y0, float _y1, float _z0, float _z1, float _k, material* m,bool _flipface = false)
        :y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k),mp(m),flipface(_flipface) {};

    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const override;
public:
    material* mp;
    bool flipface;
    float y0, y1, z0, z1, k;

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
    rec.normal = vec3(0, 0, 1);
    if (this->flipface == true) { rec.normal = vec3(0, 0, -1); }
    rec.p = r.point_at_parameter(t);
    return true;
}

bool xzrect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.mat_ptr = mp;
    rec.normal = vec3(0, 1, 0);
    if (this->flipface == true) { rec.normal = vec3(0, -1, 0); }
    rec.p = r.point_at_parameter(t);
    return true;
}
bool yzrect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    auto z = r.origin().z() + t * r.direction().z();
    auto y = r.origin().y() + t * r.direction().y();
    if (z < z0 || z > z1 || y < y0 || y > y1)
        return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    rec.normal = vec3(1, 0, 0);

    if (this->flipface == true) { rec.normal = vec3(-1, 0, 0); }
    rec.mat_ptr = mp;
    rec.p = r.point_at_parameter(t);
    return true;
}