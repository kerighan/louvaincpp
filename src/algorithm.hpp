#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using CommunityWeight = std::unordered_map<int64_t, float>;
using Connectivity = std::vector<std::vector<uint32_t>>;

CommunityWeight neighcom(
    const Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C,
    uint32_t node);

std::tuple<CommunityWeight, std::vector<float>, float> get_sigma(
    Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C);

bool move_nodes(
    Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C);

std::tuple<Connectivity, std::unordered_map<uint32_t, int64_t>> renumber(std::unordered_map<uint32_t, int64_t> &C, size_t n_nodes);

std::tuple<Connectivity, std::vector<std::vector<float>>> induced_graph(
    const Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    const Connectivity &cm2nodes,
    std::unordered_map<uint32_t, int64_t> &node2cm);

std::tuple<Connectivity,
           std::vector<std::vector<float>>,
           std::unordered_map<uint32_t, int64_t>,
           Connectivity,
           bool>
one_level(
    Connectivity &connectivity,
    std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C);

float modularity(
    std::unordered_map<uint32_t, int64_t> &C,
    Connectivity &connectivity,
    std::vector<std::vector<float>> &weights);