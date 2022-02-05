#define POWITACQ_IMPLEMENTATION 1
#include "powitacq.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
using namespace powitacq;
namespace py = pybind11;

class SpecBRDFClass {
public:
    SpecBRDFClass(const std::string & brdf_name) : brdf_name(brdf_name) {
        sbrdf = new BRDF(brdf_name);
    }
    void changeBRDF(const std::string & brdf_name) {
        this->brdf_name = brdf_name;
        delete  this->sbrdf;
        this->sbrdf = new BRDF(brdf_name);
    }

    py::array_t<float> getWavelength(){
        Spectrum lambda = this->sbrdf->wavelengths();
        int numSpec = lambda.size();

        auto spec_lambda = py::array_t<int>(numSpec);
        py::buffer_info spec_lambda_buf = spec_lambda.request();
        int * spec_lambda_ptr = (int*)spec_lambda_buf.ptr;

        for (int i = 0; i < numSpec; i++)
        {
            spec_lambda_ptr[i] = lambda[i];
        }
        return  spec_lambda;

    }

    py::array_t<float> seval(py::array_t<float>& wi, py::array_t<float>& wo)
    {
        py::buffer_info b_wi = wi.request();
        py::buffer_info b_wo = wo.request();
        float* p_wi = (float*)b_wi.ptr;
        float* p_wo = (float*)b_wo.ptr;

        Vector3f v_wi = normalize(Vector3f(p_wi[0], p_wi[1], p_wi[2]));
        Vector3f v_wo = normalize(Vector3f(p_wo[0], p_wo[1], p_wo[2]));

        Spectrum fr = this->sbrdf->eval(v_wi, v_wo);
        int numSpec = fr.size();
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
PYBIND11_MODULE(SpecBRDF, m) {
    py::class_<SpecBRDFClass>(m, "SpecBRDFClass")
            .def(py::init<const std::string &>())
            .def("eval", &SpecBRDFClass::seval)
            .def("getWavelength", &SpecBRDFClass::getWavelength)
            .def("changeBRDF", &SpecBRDFClass::changeBRDF);

    py::class_<Vector3f>(m, "vec3f")
            .def(py::init<>());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
POWITACQ_NAMESPACE_END