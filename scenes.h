#pragma once

//Many Spheres
/*int nx = 500;
    int ny = 250;
    int ns = 100;
    vec3 lookfrom = vec3(13, 2, 5);
    vec3 lookat = vec3(0, 0, 0);
    float r = (lookfrom - lookat).length();
    float radians = 1.26;
    vec3 vup = vec3(0, 1, 0);
    float fvov_degrees = 20.f;
    float aperture = 0.2;*/

__global__ void create_world1(hitable** d_list, hitable** d_world,
    camera** d_camera, int aspect, vec3 lookfrom, vec3 lookat, vec3 vup, float fvov, float aperture, curandState* rand_state) {
    curandState local_rand_state = *rand_state;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, .5, -1), 0.5,
            new lambertian(vec3(0.7, 0.3, 0.3)));

        d_list[0] = new moving_sphere(vec3(0, .5, -1),
            vec3(0, 1.0, -1), 0., 1., .5,
            new lambertian(vec3(0.7, 0.3, 0.3)));

        d_list[1] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[2] = new sphere(vec3(1, .5, -1), 0.5,
            new metal(vec3(0.8, 0.6, 0.2), 1.0));

        //hollow sphere
        d_list[3] = new sphere(vec3(-1, .5, -1), 0.5,
            new dielectric(1.5));
        d_list[4] = new sphere(vec3(-1, .5, -1), -0.45,
            new dielectric(1.5));

        int i = 1;
        for (int a = -5; a < 10; a++) {
            for (int b = -5; b < 10; b++) {
                vec3 center = vec3(-a - RND, .2, b + RND);
                d_list[4 + i] = new sphere(center, 0.2,
                    new metal(vec3(0.1f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                i++;
            }
        }

        *d_world = new hitable_list(d_list, 201);
        float focus_dist = (lookfrom - lookat).length();
        *d_camera = new camera(fvov, aspect, lookfrom, lookat, vup, aperture, focus_dist, 0., 1.);
    }
}

__global__ void free_world1(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < (201); i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

//one sphere camera near ground
__global__ void create_world(hitable** d_list, hitable** d_world,
    camera** d_camera, int aspect, vec3 lookfrom, vec3 lookat, vec3 vup, float fvov, float aperture, curandState* rand_state) {
    curandState local_rand_state = *rand_state;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, .5, -1), 0.5,
            new lambertian(vec3(0.7, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        *d_world = new hitable_list(d_list, 2);
        float focus_dist = (lookfrom - lookat).length();
        *d_camera = new camera(fvov, aspect, lookfrom, lookat, vup, aperture, focus_dist, 0., 1.);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
    for (int i = 0; i < (2); i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}