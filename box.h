#pragma once
//a box is an array of 6 sides

#include "aarect.h"

__device__ hitable* box(vec3 p, vec3 q, material* mat) {
	hitable* l[5];
	l[0] = new xyrect(p.x(),q.x(),p.y(),q.y(),p.z(),mat),
	l[1] = new xyrect(p.x(), q.x(), p.y(), q.y(), q.z(), mat),

	l[2] = new xzrect(p.x(), q.x(), p.z(), q.z(), p.y(), mat),
	l[3] = new xzrect(p.x(), q.x(), p.z(), q.z(), q.y(), mat),

	l[4] = new yzrect(p.y(), q.y(), p.z(), q.z(), p.x(), mat),
	l[5] = new yzrect(p.y(), q.y(), p.z(), q.z(), q.x(), mat);
}