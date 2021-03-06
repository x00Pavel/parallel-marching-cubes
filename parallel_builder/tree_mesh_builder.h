/**
 * @file    tree_mesh_builder.h
 *Pavel Yadlouski <xyadlo00@stud.fit.vutbr.cz>FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    unsigned octreeDevider(const ParametricScalarField &field, const Vec3_t<float> &cubeOffset, const unsigned gridSize);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
    bool blockEmpty();

    const unsigned mCutOff = 1;
	std::vector<Triangle_t> mTriangles;

};

#endif // TREE_MESH_BUILDER_H
