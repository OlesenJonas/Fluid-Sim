# Fluid-Sim

Fluid Simulation on the GPU based on the Navier-Stokes equations for incompressible flow.

## Setup
OpenGL 4.5

C++ 20
(Builds with Clang and MSVC on Windows using CMake)

vcpkg for dependencies (currently glfw + glm)


https://user-images.githubusercontent.com/45714731/186457776-b890af08-02d7-4353-bf22-6961790abb73.mp4

## Features

Velocity, Dust/Smoke Density, Temperature

### Advection
- Semi-Lagrangian (reverse) advection
  - Euler or 4th order Runge-Kutte
- Optionally: Back and forth error compensation and correction

### Buoyancy and Gravity

### Pressure Projection
- Basic Jacobi
- Basic Red-Black Gauss-Seidel
- Multigrid Jacobi
  - Only very basic restriction/interpolation schemes so far

### Rendering
- Brute Force Raymarcher following: https://shaderbits.com/blog/creating-volumetric-ray-marcher

## TODO
- Try using Gauss-Seidel iterations as smoother for the multigrid solver
- Expose more parameters to the UI (especially Buoyancy/Gravity) 
- Stagger divergence and pressure attributes on grid
- optimize everything!
  - shaders currently follow forumlas closely so its easier to understand but that also means theres big room for improvement
  - also test using groupshared memory for most passes since everything seems to be texture fetch bound
- ...
