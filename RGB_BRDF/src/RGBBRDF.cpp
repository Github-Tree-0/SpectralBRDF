#define POWITACQ_IMPLEMENTATION 1
#include "powitacq_rgb.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using namespace powitacq_rgb;
namespace py = pybind11;

class RGBBRDFClass {
public:
    RGBBRDFClass(const std::string & brdf_name) : brdf_name(brdf_name) {
        sbrdf = new BRDF(brdf_name);
    }
    void changeBRDF(const std::string & brdf_name) {
        this->brdf_name = brdf_name;
        delete this->sbrdf;
        this->sbrdf = new BRDF(brdf_name);
    }

    py::array_t<float> seval(py::array_t<float>& wi, py::array_t<float>& wo)
    {
        py::buffer_info b_wi = wi.request();
        py::buffer_info b_wo = wo.request();
        float* p_wi = (float*)b_wi.ptr;
        float* p_wo = (float*)b_wo.ptr;

        Vector3f v_wi = normalize(Vector3f(p_wi[0], p_wi[1], p_wi[2]));
        Vector3f v_wo = normalize(Vector3f(p_wo[0], p_wo[1], p_wo[2]));

        Vector3f fr = this->sbrdf->eval(v_wi, v_wo);
        int numSpec = 3;
        auto spec_response = py::array_t<float>(numSpec);
        py::buffer_info spec_response_buf = spec_response.request();
        float * spec_response_ptr = (float*)spec_response_buf.ptr;

        for (int i = 0; i < numSpec; i++)
        {
            spec_response_ptr[i] = fr[i];
        }

        return  spec_response;
    }

private:
    std::string brdf_name;
    BRDF* sbrdf;
};



POWITACQ_NAMESPACE_BEGIN
//using namespace powitacq_rgb;
PYBIND11_MODULE(RGBBRDF, m) {
    py::class_<RGBBRDFClass>(m, "RGBBRDFClass")
            .def(py::init<const std::string &>())
            .def("eval", &RGBBRDFClass::seval)
            .def("changeBRDF", &RGBBRDFClass::changeBRDF);

    py::class_<Vector3f>(m, "vec3f")
            .def(py::init<>());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
POWITACQ_NAMESPACE_END