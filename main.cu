#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "moving_sphere.h"
#include "aarect.h"
#include <fstream>
#include <string>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
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

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, vec3& background) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    hit_record rec;
    //if the initially fired ray hits nothing return background color bblack
    if (!(*world)->hit(r, 0.001f, FLT_MAX, rec)) {

        return background;
    }

    for (int i = 0; i < 50; i++) {


        
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            vec3 emitted = rec.mat_ptr->emmitted(rec.u, rec.v, rec.p);
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                //return rec.mat_ptr->emmitted(rec.u, rec.v, rec.p);
                //return cur_attenuation;
                return cur_attenuation* rec.mat_ptr->emmitted(rec.u, rec.v, rec.p);
         

            }
        }


        else {
            
            return cur_attenuation;
        }

    }
    return background; // exceeded recursion
}


__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1.984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world,curandState* rand_state, vec3 background) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state, background);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col = clamp(vec3(0, 0, 0),vec3(1, 1, 1), col);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}
#define RND (curand_uniform(&local_rand_state))

/*SET WORLD*/

__global__ void create_world(hitable** d_list, hitable** d_world,
    camera** d_camera, int aspect, vec3 lookfrom, vec3 lookat, vec3 vup, float fvov, float aperture, curandState* rand_state) {
    curandState local_rand_state = *rand_state;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //my camera settings currently have y direction going up

        d_list[0] = new xyrect(0, 555., 0, 555., 0, new lambertian(vec3(0,.7,0))); //left

        d_list[1] = new yzrect(0, 555., 0, 555., 555, new lambertian(vec3(.7, .7, .7)),true); //back
        
        material* m = new diffuse_light(vec3(15,15,15));
        d_list[2] = new xzrect(0,555., 0, 555.,0, new lambertian(vec3(.7,.7,.7))); //bottom
        d_list[3] = new xyrect(0, 555., 0, 555., 555, new lambertian(vec3(1, 0, 0)),true); //right
        d_list[4] = new xzrect(0, 555., 0, 555., 555, new lambertian(vec3(.7, .7, .7)),true); //top
        d_list[5] = new xzrect(213, 343, 227, 332, 554, m); //light

        *d_world = new hitable_list(d_list, 6);
        float focus_dist = (lookfrom - lookat).length();
        *d_camera = new camera(fvov, aspect, lookfrom, lookat, vup, aperture, focus_dist, 0., 1.);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < (6); i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}






//takes in image width,height,numsamples,lookfrom,lookat,vup,fvov_degrees,aperture
void run(int nx, int ny, int ns, vec3 lookfrom, vec3 lookat, vec3 vup, float fvov_degrees, float aperture,std::string filename, vec3& background) {



    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // make our world of hitables & the camera
    hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 22 * 22 * sizeof(hitable*)));
    hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    camera** d_camera;


    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, int(nx / ny), lookfrom, lookat, vup, fvov_degrees, aperture, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world,d_rand_state,background);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::ofstream myfile;

    myfile.open(filename);
    myfile << "P3\n" << nx << ' ' << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            myfile << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
int main() {
    //takes in image width,height,numsamples,lookfrom,lookat,vup,fvov_degrees,aperture
    int nx = 500;
    int ny = 500;
    int ns = 100;
    vec3 lookfrom = vec3(-800,278,278);
    vec3 lookat = vec3(0,278,278);
    vec3 background = vec3(0, 0, 0);
    float r = (lookfrom - lookat).length();
    float radians = 1.26;
    vec3 vup = vec3(0, 1, 0);
    float fvov_degrees = 40.;
    float aperture = 0.2;

    //make a video 30 fps for 5secs so 150 frames total
    //each frame the camera will rotate 360/150 = 2.4 degrees around the or 0.0418879 radians
    //each image will have in incremental name
    for (int i = 0; i < 1; ++i) {
        std::string filename = "image" + std::to_string(i);
        filename = filename + ".ppm";
        run(nx, ny, ns, lookfrom, lookat, vup, fvov_degrees, aperture, filename,background);
        lookfrom -= vec3(0, 0, 1);
        radians += 1.26;
    }

}