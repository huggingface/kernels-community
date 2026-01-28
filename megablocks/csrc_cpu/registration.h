/*****************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 ****************************************************************************************/

// MegaBlocks CPU Registration Header

#pragma once

#include <Python.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)           \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() { \
    static struct PyModuleDef module = {   \
        PyModuleDef_HEAD_INIT,             \
        STRINGIFY(NAME),                   \
        nullptr,                           \
        0,                                 \
        nullptr,                           \
        nullptr,                           \
        nullptr,                           \
        nullptr,                           \
        nullptr};                          \
    return PyModule_Create(&module);       \
  }
