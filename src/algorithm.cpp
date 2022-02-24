#include <iostream>
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
    uint32_t node)
{
    CommunityWeight neighbor_weight;
    std::vector<uint32_t> neighbors = connectivity[node];
    std::vector<float> neighbors_weight = weights[node];
    for (size_t i = 0; i < neighbors.size(); i++)
    {
        uint32_t neighbor = neighbors[i];
        if (neighbor == node)
            continue;

        int64_t neighbor_c = C[neighbor];
        float weight = neighbors_weight[i];
        neighbor_weight[neighbor_c] += weight;
    }
    return neighbor_weight;
}

std::tuple<CommunityWeight, std::vector<float>, float> get_sigma(
    Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C)
{
    std::vector<float> node_degree;

    CommunityWeight sigma;
    float m = 0;
    for (uint32_t u = 0; u < connectivity.size(); u++)
    {
        std::vector<uint32_t> neighbors = connectivity[u];
        std::vector<float> neighbors_weight = weights[u];
        float degree = 0;
        for (size_t i = 0; i < neighbors.size(); i++)
        {
            uint32_t v = neighbors[i];
            int64_t v_c = C[v];
            float weight = neighbors_weight[i];
            degree += weight;
            m += weight;
            sigma[v_c] += weight;
        }
        node_degree.push_back(degree);
    }
    return std::make_tuple(sigma, node_degree, m);
}

bool move_nodes(
    Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C)
{

    auto [sigma, node_degree, m] = get_sigma(connectivity, weights, C);
    int n_changes = -1;
    bool modified = false;
    while (n_changes != 0)
    {
        n_changes = 0;
        for (uint32_t node = 0; node < connectivity.size(); node++)
        {
            int64_t node_c = C[node];
            float node_d = node_degree[node];

            CommunityWeight neighbor_weight = neighcom(
                connectivity, weights, C, node);

            // remove
            C[node] = -1;
            sigma[node_c] -= node_d;

            int64_t best_c = node_c;
            float best_q = 0.;
            for (auto &x : neighbor_weight)
            {
                int64_t c = x.first;
                float weight = x.second;

                if (weight <= 0)
                    continue;

                float delta_q = weight - node_d * sigma[c] / m;
                if (delta_q > best_q)
                {
                    best_q = delta_q;
                    best_c = c;
                }
            }

            // insert
            C[node] = best_c;
            sigma[best_c] += node_d;

            if (best_c != node_c)
            {
                n_changes++;
                modified = true;
            }
        }
    }
    return modified;
}

std::tuple<Connectivity, std::unordered_map<uint32_t, int64_t>> renumber(std::unordered_map<uint32_t, int64_t> &C, size_t n_nodes)
{
    std::vector<int64_t> com_n_nodes;
    com_n_nodes.resize(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++)
    {
        com_n_nodes[C[i]] += 1;
    }

    std::vector<int64_t> com_new_index;
    com_new_index.resize(n_nodes);
    int64_t final_index = 0;
    for (size_t com = 0; com < n_nodes; com++)
    {
        if (com_n_nodes[com] <= 0)
            continue;
        com_new_index[com] = final_index;
        final_index++;
    }

    Connectivity new_communities;
    new_communities.resize(final_index);

    std::unordered_map<uint32_t, int64_t> new_node2com;
    for (uint32_t node = 0; node < n_nodes; node++)
    {
        int64_t new_index = com_new_index[C[node]];
        new_communities[new_index].push_back(node);
        new_node2com[node] = new_index;
    }
    return std::make_tuple(new_communities, new_node2com);
}

std::tuple<Connectivity, std::vector<std::vector<float>>> induced_graph(
    const Connectivity &connectivity,
    const std::vector<std::vector<float>> &weights,
    const Connectivity &cm2nodes,
    std::unordered_map<uint32_t, int64_t> &node2cm)
{
    size_t new_n_nodes = cm2nodes.size();
    Connectivity new_connectivity;
    new_connectivity.resize(new_n_nodes);

    std::vector<std::vector<float>> new_weights;
    new_weights.resize(new_n_nodes);

    std::unordered_map<int64_t, float> to_insert;
    for (size_t com = 0; com < new_n_nodes; com++)
    {
        to_insert.clear();
        std::vector<uint32_t> nodes = cm2nodes[com];
        for (size_t z = 0; z < nodes.size(); z++)
        {
            uint32_t node = nodes[z];
            std::vector<uint32_t> neighbors = connectivity[node];
            std::vector<float> neighbors_weight = weights[node];

            for (size_t i = 0; i < neighbors.size(); i++)
            {
                uint32_t neighbor = neighbors[i];
                float neighbor_w = neighbors_weight[i];
                int64_t neighbor_c = node2cm[neighbor];
                if (neighbor == node)
                {
                    to_insert[neighbor_c] += 1 * neighbor_w;
                }
                else
                {
                    to_insert[neighbor_c] += neighbor_w;
                }
            }
        }
        for (auto [key, value] : to_insert)
        {
            new_connectivity[com].push_back(key);
            if (key == com)
            {
                new_weights[com].push_back(value / 1.);
            }
            else
            {
                new_weights[com].push_back(value);
            }
        }
    }
    return std::make_tuple(new_connectivity, new_weights);
}

std::tuple<Connectivity,
           std::vector<std::vector<float>>,
           std::unordered_map<uint32_t, int64_t>,
           Connectivity,
           bool>
one_level(
    Connectivity &connectivity,
    std::vector<std::vector<float>> &weights,
    std::unordered_map<uint32_t, int64_t> &C)
{
    bool modified = move_nodes(connectivity, weights, C);
    auto [cm2nodes, new_C] = renumber(C, connectivity.size());
    auto [cn, wg] = induced_graph(
        connectivity, weights, cm2nodes, new_C);

    return std::make_tuple(cn, wg, new_C, cm2nodes, modified);
}

float modularity(
    std::unordered_map<uint32_t, int64_t> &C,
    Connectivity &connectivity,
    std::vector<std::vector<float>> &weights)
{
    std::vector<float> degrees;
    float links = 0.;
    for (uint32_t node = 0; node < connectivity.size(); node++)
    {
        float degree = 0.;
        std::vector<uint32_t> neighbors = connectivity[node];
        std::vector<float> neighbors_weight = weights[node];
        for (size_t i = 0; i < neighbors.size(); i++)
        {
            uint32_t neighbor = neighbors[i];
            if (neighbor == node)
            {
                degree += 2 * neighbors_weight[i];
                links += 2 * neighbors_weight[i];
            }
            else
            {
                degree += neighbors_weight[i];
                links += neighbors_weight[i];
            }
        }
        degrees.push_back(degree);
    }
    links /= 2;

    std::unordered_map<int64_t, float> inc;
    std::unordered_map<int64_t, float> deg;
    for (uint32_t node = 0; node < connectivity.size(); node++)
    {
        int64_t com = C[node];
        deg[com] += degrees[node];

        std::vector<uint32_t> neighbors = connectivity[node];
        std::vector<float> neighbors_weight = weights[node];
        for (size_t i = 0; i < neighbors.size(); i++)
        {
            uint32_t neighbor = neighbors[i];
            if (C[neighbor] == com)
            {
                if (neighbor == node)
                {
                    inc[com] += neighbors_weight[i];
                }
                else
                {
                    inc[com] += neighbors_weight[i] / 2;
                }
            }
        }
    }

    float res = 0.;
    for (auto [com, val] : deg)
    {
        float scale = val / (2. * links);
        scale *= scale;
        res += (inc[com] / links) - scale;
    }
    return res;
}