#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algorithm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_louvaincpp, m)
{
    m.def("get_sigma", &get_sigma, "get sigma tot");
    m.def("neighcom", &neighcom, "neighbors communities");
    m.def("move_nodes", &move_nodes, "move nodes");
    m.def("renumber", &renumber, "renumber");
    m.def("induced_graph", &induced_graph, "induced graph");
    m.def("one_level", &one_level, "one_level");
    m.def("modularity", &modularity, "modularity");
}
