#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "utility.h"
#include "color.h"
#include "hittable_list.h"
#include "sphere.h"

#include <iostream>
#include <fstream>
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__
float hit_sphere(const vec3 center, float radius, const ray r) { 
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0) {
        return -1.0;
    }
    else {
        return (-b - sqrt(disc)) / (2.0 * a);
    }
}


__device__ 
vec3 color_ray(const ray& r,hittable** world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

__global__
void render(vec3* fb,  int image_width,  int image_height,
    vec3 o, vec3 h, vec3 vert, vec3 l, hittable** world) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row>=image_height || col >=image_width)return;
    int pindex = (row * image_width + col);
    float u = float(col) / float(image_width);
    float v = float(row) / float(image_height);
    ray r(o, l + u * h + v * vert);
    fb[pindex] = color_ray(r,world);
}

__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
}

int main() {

    // Image
    float aspect_ratio = 16.0f / 9.0f;

    const int image_width = 500;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    //Camera
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    //Allocate Frame buffer and determine  bytesize
    size_t byte_size = 3*image_width * image_height * sizeof(vec3);
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**) & fb, byte_size));

    //determine block_size
    int block_width = 16;
    int block_height = 16;

    //determine gridsize and assign to dim3
    int grid_width = image_width / block_width + 1;
    int grid_height = image_height / block_height + 1;

    dim3 blocks(grid_width, grid_height);
    dim3 threads(block_width, block_height);

    // make our world of hitables
    hittable **d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //render
    render<<<blocks, threads >>> (fb, 
        image_width, 
        image_height,
        origin,
        horizontal,
        vertical,
        lower_left_corner,
        d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // output
    std::ofstream myfile;
    myfile.open("image.ppm");
    myfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;

        for (int i = 0; i < image_width; ++i) {
            size_t pindex = (j * image_width + i);

            float r = fb[pindex].x();
            float g = fb[pindex].y();
            float b = fb[pindex].z();
            write_color(myfile, fb[pindex]);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();

}

