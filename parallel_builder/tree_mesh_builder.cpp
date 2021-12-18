/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Pavel Yadlouski <xyadlo00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}


unsigned TreeMeshBuilder::octreeDevider(const ParametricScalarField &field, const Vec3_t<float> &cubeOffset, const unsigned gridSize){
    
    unsigned totalTriangles = 0;
    // check if it is already a time to stop (gridSize < mCutOff) the devision
    // and run buildCube
    if (gridSize == mCutOff){
        unsigned tmp = buildCube(cubeOffset, field);

        #pragma omp atomic update
        totalTriangles += tmp;
    }
    else{
        const float realEdgeSize = gridSize * mGridResolution;
        const float rightSideCondition = field.getIsoLevel() + (sqrt(3.0f)/2.0f)*realEdgeSize;
        const float halfEdge = gridSize/2.0f;

        const Vec3_t<float> midCubeOffset = {
            (cubeOffset.x + halfEdge) * mGridResolution ,
            (cubeOffset.y + halfEdge) * mGridResolution ,
            (cubeOffset.z + halfEdge) * mGridResolution };

        const float leftSideCondition = evaluateFieldAt(midCubeOffset, field);
        // check that current block is empty
        if (leftSideCondition > rightSideCondition){
            return 0;
        }
        
        // block is not empty, so creating tasks for each field in the vector
        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index1 = {
                cubeOffset.x,
                cubeOffset.y,
                cubeOffset.z
                };
            unsigned tmp = octreeDevider(field, index1, halfEdge); 

            #pragma omp atomic update
            totalTriangles += tmp;
        }
        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index2 = {
                cubeOffset.x + halfEdge,
                cubeOffset.y,
                cubeOffset.z
                };

            unsigned tmp = octreeDevider(field, index2, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }
        #pragma omp task shared(field, halfEdge, totalTriangles)
        {        
            const Vec3_t<float> index3 = {
                cubeOffset.x,
                cubeOffset.y + halfEdge,
                cubeOffset.z 
                };
            
            unsigned tmp = octreeDevider(field, index3, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }

        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index4 = {
                cubeOffset.x,
                cubeOffset.y,
                cubeOffset.z + halfEdge
                };

            unsigned tmp = octreeDevider(field, index4, halfEdge);
            #pragma omp atomic update
            totalTriangles += tmp;
        }

        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index5 = {
                cubeOffset.x + halfEdge,
                cubeOffset.y + halfEdge,
                cubeOffset.z
                };

            unsigned tmp = octreeDevider(field, index5, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }

        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index6 = {
                cubeOffset.x + halfEdge,
                cubeOffset.y,
                cubeOffset.z + halfEdge
                };
            unsigned tmp = octreeDevider(field, index6, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }

        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index7 = {
                cubeOffset.x,
                cubeOffset.y + halfEdge,
                cubeOffset.z + halfEdge
                };
            unsigned tmp = octreeDevider(field, index7, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }

        #pragma omp task shared(field, halfEdge, totalTriangles)
        {
            const Vec3_t<float> index8 = {
                cubeOffset.x + halfEdge,
                cubeOffset.y + halfEdge,
                cubeOffset.z + halfEdge
                };
            unsigned tmp = octreeDevider(field, index8, halfEdge);

            #pragma omp atomic update
            totalTriangles += tmp;
        }
    }
    
    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    
    unsigned totalTriangles = 0;
    #pragma omp parallel default(none) shared(totalTriangles, field)
    #pragma omp  single
    totalTriangles = octreeDevider(field, Vec3_t<float>(), mGridSize);

    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(const Vec3_t<float> point : field.getPoints())
    {
        float distanceSquared  = (pos.x - point.x) * (pos.x - point.x);
        distanceSquared       += (pos.y - point.y) * (pos.y - point.y);
        distanceSquared       += (pos.z - point.z) * (pos.z - point.z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    #pragma omp critical(treeEmitTriangle)
    mTriangles.push_back(triangle);
}
