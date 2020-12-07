#pragma once
#include "ray.h"

class aabb {
public:
	__device__ aabb() {}
	__device__ aabb(const vec3& a, const vec3& b) { _min = a; _max = b; }
	__device__ aabb(aabb box0, aabb box1) {
	
		vec3 small(fmin(box0.min().x(), box1.min().x()),
			fmin(box0.min().y(), box1.min().y()),
			fmin(box0.min().z(), box1.min().z()));

		vec3 big(fmin(box0.max().x(), box1.max().x()),
			fmin(box0.max().y(), box1.max().y()),
			fmin(box0.max().z(), box1.max().z()));
		_min = small;
		_max = big;
	}
	__device__ vec3 min() { return _min; }
	__device__ vec3 max() { return _max; }

	//to hit the bounding box the ray must hit each slab x,y,z
	__device__ bool hit(const ray& r, float tmin, float tmax) const {
		for (int x = 0; x < 3; x++) {
			float t0 = fmin((_min[x] - r.origin()[x]) / r.direction()[x],
				(_max[x] - r.origin()[x] / r.direction()[x]));
			float t1 = fmax((_min[x] - r.origin()[x]) / r.direction()[x],
				(_max[x] - r.origin()[x] / r.direction()[x]));
			tmin = fmax(t0, tmin);
			tmax = fmin(t1, tmax);
			if (tmax <= tmin) { return false; }
		}
		return true;
	
	}

public:
	vec3 _min;
	vec3 _max;

};