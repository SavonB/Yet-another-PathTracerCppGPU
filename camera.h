#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}
class camera {
public:
    
    
    __device__ camera(float vfov, float aspect,
        vec3 lookfrom, vec3 lookat, vec3 vup) {
        float theta = 3.14159 / 180 * vfov;
        float h = tan(theta / 2);

        float vph = 2.f * h;
        float vpw = aspect * vph;
        vec3 w = unit_vector(lookfrom - lookat);
        vec3 u = unit_vector(cross(vup, w));
        vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = vpw * u;
        vertical = vph * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;

    }
    __device__ camera(float vfov, float aspect,
        vec3 lookfrom, vec3 lookat, vec3 vup,float aperture,float focus_dist) {
        float theta = 3.14159 / 180 * vfov;
        float h = tan(theta / 2);

        float vph = 2.f * h;
        float vpw = aspect * vph;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = vpw * u*focus_dist;
        vertical = vph * v * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
        lens_radius = aperture / 2;
    }

    __device__ ray get_ray(float s, float t,
        curandState* local_rand_state)
    { 
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin+offset, lower_left_corner + s * horizontal + t * vertical - origin-offset); }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif