// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "stop_sign.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "static_object.h"

#include <memory>

namespace py = pybind11;

namespace nocturne {

void DefineStopSign(py::module& m) {
  py::class_<StopSign, std::shared_ptr<StopSign>>(m, "StopSign")

      .def_property_readonly("type", &StopSign::Type)
      .def("position", &StopSign::position);
}

}  // namespace nocturne