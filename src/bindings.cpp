#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algorithm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_louvaincpp, m)
{
    m.def("get_adj", &get_adj);
    m.def("init_status", &init_status);
    m.def("neighcom", &neighcom);
    m.def("modularity", &modularity);
    m.def("one_level", &one_level);
    m.def("renumber", &renumber);
    m.def("induced_graph", &induced_graph);
    m.def("get_partition", &get_partition);
    m.def("generate_dendrogram", &generate_dendrogram);
    m.def("generate_full_dendrogram", &generate_full_dendrogram);
}
