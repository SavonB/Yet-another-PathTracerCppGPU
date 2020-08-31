#pragma once
#include "vec3.h"
#include <curand_kernel.h>

class custom_texture{
public:
    __device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class solid_color : custom_texture {
public:
    __device__ solid_color() {}
    __device__ solid_color(vec3 c) : color_value(c) {}

    __device__ solid_color(float r, float g, float b) : color_value(vec3(r, g, b)) {}

    __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
        return color_value;
    
    }
private:
    vec3 color_value;

};