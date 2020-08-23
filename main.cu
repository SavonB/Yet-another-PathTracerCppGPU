#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <iostream>
#include <fstream>
#include "color.h"
#include "vec3.h"

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

__global__
void render(vec3* fb,  int image_width,  int image_height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row>=image_height || col >=image_width)return;
    int pindex = (row * image_width + col);
    fb[pindex] = vec3(float(col) / image_width, float(row) / image_height,.25);
}

int main() {

    // Image

    const int image_width = 500;
    const int image_height = 500;


    //Allocate Frame buffer and determine  bytesize
    size_t byte_size = 3*image_width * image_height * sizeof(vec3);
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**) & fb, byte_size));

    //determine block_size
    int block_width = 8;
    int block_height = 8;

    //determine gridsize and assign to dim3
    int grid_width = image_width / block_width + 1;
    int grid_height = image_height / block_height + 1;

    dim3 blocks(grid_width, grid_height);
    dim3 threads(block_width, block_height);

    //render
    render<<<blocks, threads >>> (fb, image_width, image_height);
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

            auto r = fb[pindex].x();
            auto g = fb[pindex].y();
            auto b = fb[pindex].z();
            write_color(myfile, fb[pindex]);
        }
    }
    cudaFree(fb);
}

