#pragma once
#include "vec3.h"

#include <iostream>

void write_color(std::ostream& myfile, color pixel_color) {
	int ir = static_cast<int>(255.999 * pixel_color.x());
	int ig = static_cast<int>(255.999 * pixel_color.y());
	int ib = static_cast<int>(255.999 * pixel_color.z());

	myfile << ir << ' ' << ig << ' ' << ib << '\n';
}