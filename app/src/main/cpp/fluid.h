//
// Created by joesc on 12/6/2021.
//



#ifndef FLUIDSIM_FLUID_H
#define FLUIDSIM_FLUID_H

#define Fluid3d

#include "linalg.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <EGL/egl.h>
#include <pthread.h>
#include <thread>

struct Particle;
using namespace std;
typedef unsigned int uint;
typedef const uint cuint;
// https://matthias-research.github.io/pages/publications/tetraederCollision.pdf

#ifdef Fluid3d
struct hash_func : unary_function<ivec2, size_t> {
    size_t operator()(ivec2 const& pos) const {
        return(size_t) ((pos.x * 73856093) ^ (pos.y * 19349663) );//^ (pos.z * 83492791));
    }
};
typedef unordered_map<ivec2,vector<Particle*>,hash_func> hmap_t;
struct Hasher{
    hmap_t map;
    vector<ivec2> offs;
    GLfloat sizeInv;

    Hasher(cuint numBuckets, GLfloat cellSize) : map(numBuckets), sizeInv(1.f/cellSize){
        for(int i=-1; i<=1; i++)
            for(int j=-1; j<=1; j++)
                //for(int k=-1;k<1;k++)
                offs.push_back(ivec2(i, j));
    }
    void insert(const vec2& pos, Particle* p ){map[toBucket(pos)].push_back(p);}
    void clear(){map.clear();}
    inline ivec2 toBucket(const vec2& pos)const {return ivec2(floor(pos*sizeInv));}

    void make_nlist(vec2& pos, vector<Particle*>& out) const {
        const ivec2 bkt = toBucket(pos);
        for(const auto& o : offs){
            auto it = map.find(o+bkt);
            if(it != map.end())
                out.insert(out.end(), it->second.begin(), it->second.end());
        }
    }
};
struct Neighbor{
    Particle* j;
    GLfloat dist, q, q2;
};
struct Particle{
    vec4 col;
    vec2 dens, press, visc;

    vec2 pos,pos_old, vel, force;
    vector<Neighbor> neighbors;
};
struct Fluid{
    struct{
        int w, h;
        const EGLContext ctx;
    }window;
    vector<Particle> particles;
    vec2 gravity_dir;
    Fluid():window({0,0,eglGetCurrentContext()}),gravity_dir(0.0,-1.0){}
    void resize(int w, int h);
    void display();
    bool init(cuint);
    void step();
    void update_gravity(float theta){
        gravity_dir = vec2(sin(theta),-cos(theta)); // don't no why updating gravity
    }
    void step_prep(int i);
    float cubickernel(float r, float h);
    float cubickernelgrad(float r, float h);
    void step_density(int i);
    void step_pressure(int i);
    void step_pressure_force(int i);
    void step_viscosity(int i);
    void step_color(int i);
};
Fluid* make_fluid(const unsigned int N);
#else

struct hash_func : unary_function<ivec2, size_t> {
    size_t operator()(ivec2 const& pos) const {
        return(size_t) ((pos.x * 73856093) ^ (pos.y * 19349663));
    }
};
typedef unordered_map<ivec2,vector<Particle*>,hash_func> hmap_t;
struct Hasher{
    hmap_t map;
    vector<ivec2> offs;
    GLfloat sizeInv;

    Hasher(cuint numBuckets, GLfloat cellSize) : map(numBuckets), sizeInv(1.f/cellSize){
        for(int i=-1; i<=1; i++)
            for(int j=-1; j<=1; j++)
                offs.push_back(ivec2(i, j));
    }
    void insert(const vec2& pos, Particle* p ){map[toBucket(pos)].push_back(p);}
    void clear(){map.clear();}
    inline ivec2 toBucket(const vec2& pos)const {return ivec2(floor(pos*sizeInv));}

    void make_nlist(vec2& pos, vector<Particle*>& out) const {
        const ivec2 bkt = toBucket(pos);
        for(const auto& o : offs){
            auto it = map.find(o+bkt);
            if(it != map.end())
                out.insert(out.end(), it->second.begin(), it->second.end());
        }
    }
};
struct Neighbor{
    Particle* j;
    GLfloat q, q2;
};
struct Particle{
    vec4 col;
    vec2 dens, press, visc;

    vec2 pos,pos_old, vel, force;
    vector<Neighbor> neighbors;
};
struct Fluid{
    struct{
        int w, h;
        const EGLContext ctx;
    }window;
    vector<Particle> particles;
    vec2 gravity_dir;
    Fluid():window({0,0,eglGetCurrentContext()}),gravity_dir(0.0,-1.0){}
    void resize(int w, int h);
    void display();
    bool init(cuint);
    void step();
    void update_gravity(float theta){
        gravity_dir = vec2(sin(theta),-cos(theta));
    }
    void step_prep(int i);
    void step_density(int i);
    void step_pressure(int i);
    void step_pressure_force(int i);
    void step_viscosity(int i);
    void step_color(int i);
};
Fluid* make_fluid(const unsigned int N);


#endif // Fluid3d


#endif //FLUIDSIM_FLUID_H
