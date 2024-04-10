#include <iostream>
#include <random>
#include <chrono>
#include <queue>
#include <tuple>
using namespace std;
const int LATTICE_SIDE_LENGTH = 128;
const int COUPLING_STRENGTH = 1;
const int LONGITUDINAL_FIELD_STRENGTH = 0;
const int TRANSVERSE_FIELD_STANDARD_DEVIATION = 1;

struct Parameter {
    int x1;
    int y1;
    int z1;
    double strength;
    Parameter (int x1, int y1, int z1, double strength) : x1(x1), y1(y1), z1(z1), strength(strength) {}
};
struct Node : Parameter {
    Node (int x1, int y1, int z1, double strength) : Parameter(x1, y1, z1, strength) {}
};

struct Edge : Parameter {
    int x2;
    int y2;
    int z2;
    Edge (int x1, int y1, int z1, double strength, int x2, int y2, int z2) : 
        Parameter(x1, y1, z1, strength), x2(x2), y2(y2), z2(z2) {}
};

struct CompareParameters {
    bool operator()(Parameter const& p1, Parameter const& p2) {
        return p1.strength < p2.strength;
    }
};

priority_queue<Node, vector<Node>, CompareParameters> Nodes;
priority_queue<Edge, vector<Edge>, CompareParameters> Edges;

int main() {
	// generate the network randomly using 
    // gaussian distribution for fields centered at 0
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> distribution(0, TRANSVERSE_FIELD_STANDARD_DEVIATION);
    for (int i = 0; i < LATTICE_SIDE_LENGTH; i++) {
        for (int j = 0; j < LATTICE_SIDE_LENGTH; j++) {
            for (int k = 0; k < LATTICE_SIDE_LENGTH; k++) {
                Nodes.push(Node(i, j, k, distribution(generator)));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, i, j, (k + 1) % LATTICE_SIDE_LENGTH));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, i, j, (k - 1 + LATTICE_SIDE_LENGTH) % LATTICE_SIDE_LENGTH));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, i, (j + 1) % LATTICE_SIDE_LENGTH, k));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, i, (j - 1 + LATTICE_SIDE_LENGTH) % LATTICE_SIDE_LENGTH, k));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, (i + 1) % LATTICE_SIDE_LENGTH, j, k));
                Edges.push(Edge(i, j, k, COUPLING_STRENGTH, (i - 1 + LATTICE_SIDE_LENGTH) % LATTICE_SIDE_LENGTH, j, k));
            }
        }
    }
    // run the first order rules using heap
    // unique id for each node  
    while(true) {
        if (remaining_fields == 1) {
            break;
        }
        // find the node with the largest field by iterating through the lattice
        int remaining_field = 0;
        for (int i = 0; i < LATTICE_SIDE_LENGTH; i++) {
            for (int j = 0; j < LATTICE_SIDE_LENGTH; j++) {
                for (int k = 0; k < LATTICE_SIDE_LENGTH; k++) {
                    if (fields[i][j][k] > remaining_field) {
                        remaining_field = fields[i][j][k];
                        largest_field_location = make_tuple(i, j, k);
                    }
                }
            }
        }
        // apply the first order rules to the node with the largest field
        if (remaining_field > COUPLING_STRENGTH) {
            // node decimation rules
            fields[get<0>(largest_field_location)][get<1>(largest_field_location)][get<2>(largest_field_location)] = 0;
            remaining_fields--;
            
        } else {
            // edge decimation rules

        }
        // might need to be careful about if only 2 nodes are left
        // want to use periodic boundary conditions ie wrap around the boundary
    }
    // return the field of the last remaining node
	return remaining_field;
}