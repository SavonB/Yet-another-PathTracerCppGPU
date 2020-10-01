#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};

        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;
        vec3 center;
        float radius;
        material *mat_ptr;
};

__device__ void get_sphere_uv(const vec3& p, float& u, float& v) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    u = 1 - (phi + pi) / (2 * pi);
    v = (theta + pi / 2) / pi;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}





#endif