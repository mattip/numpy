import os
import genapi

from genapi import \
        TypeApi, GlobalVarApi, FunctionApi, BoolValuesApi

import numpy_api

# use annotated api when running under cpychecker
h_template = r"""
#if defined(_MULTIARRAYMODULE) || defined(WITH_CPYCHECKER_STEALS_REFERENCE_TO_ARG_ATTRIBUTE)

typedef struct {
        PyObject_HEAD
        npy_bool obval;
} PyBoolScalarObject;

HPyType_LEGACY_HELPERS(PyBoolScalarObject)

extern NPY_NO_EXPORT PyTypeObject *PyArrayMapIter_Type;
extern NPY_NO_EXPORT PyTypeObject *PyArrayNeighborhoodIter_Type;

%s

// HPy

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API=NULL;
#endif
#endif

#if defined(HPY_ARRAY_UNIQUE_SYMBOL)
#define HPyArray_API HPY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **HPyArray_API;
#else
HPyContext *numpy_global_ctx = NULL;
#if defined(HPY_ARRAY_UNIQUE_SYMBOL)
void **HPyArray_API;
#else
static void **HPyArray_API=NULL;
#endif
#endif

%s

// HPy

%s

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
_import_array(void)
{
  int st;
  PyObject *numpy = PyImport_ImportModule("numpy.core._multiarray_umath");
  PyObject *c_api = NULL;
  PyObject *hpy_api = NULL;
  PyObject *ctx_capsule = NULL;

  if (numpy == NULL) {
      return -1;
  }
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  if (c_api == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_ARRAY_API not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(c_api)) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is not PyCapsule object");
      Py_DECREF(c_api);
      return -1;
  }
  PyArray_API = (void **)PyCapsule_GetPointer(c_api, NULL);
  Py_DECREF(c_api);
  if (PyArray_API == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_ARRAY_API is NULL pointer");
      return -1;
  }

  ctx_capsule = PyObject_GetAttrString(numpy, "_HPY_CONTEXT");
  Py_DECREF(numpy);
  if (ctx_capsule == NULL) {
      PyErr_SetString(PyExc_AttributeError, "_HPY_CONTEXT not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(ctx_capsule)) {
      PyErr_SetString(PyExc_RuntimeError, "_HPY_CONTEXT is not PyCapsule object");
      Py_DECREF(ctx_capsule);
      return -1;
  }
  numpy_global_ctx = (HPyContext *)PyCapsule_GetPointer(ctx_capsule, NULL);
  Py_DECREF(ctx_capsule);
  if (numpy_global_ctx == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "_HPY_CONTEXT is NULL pointer");
      return -1;
  }

  hpy_api = PyObject_GetAttrString(numpy, "_HPY_ARRAY_API");
  if (c_api == NULL) {
      HPyErr_SetString(numpy_global_ctx, numpy_global_ctx->h_AttributeError, "_HPY_ARRAY_API not found");
      return -1;
  }

  if (!PyCapsule_CheckExact(hpy_api)) {
      HPyErr_SetString(numpy_global_ctx, numpy_global_ctx->h_RuntimeError, "_HPY_ARRAY_API is not PyCapsule object");
      Py_DECREF(hpy_api);
      return -1;
  }

  HPyArray_API = (void **)PyCapsule_GetPointer(hpy_api, NULL);
  Py_DECREF(hpy_api);
  if (HPyArray_API == NULL) {
      HPyErr_SetString(numpy_global_ctx, numpy_global_ctx->h_RuntimeError, "_HPY_ARRAY_API is NULL pointer");
      return -1;
  }

  /* Perform runtime check of C API version */
  if (NPY_VERSION != PyArray_GetNDArrayCVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "ABI version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_VERSION, (int) PyArray_GetNDArrayCVersion());
      return -1;
  }
  if (NPY_FEATURE_VERSION > PyArray_GetNDArrayCFeatureVersion()) {
      PyErr_Format(PyExc_RuntimeError, "module compiled against "\
             "API version 0x%%x but this version of numpy is 0x%%x", \
             (int) NPY_FEATURE_VERSION, (int) PyArray_GetNDArrayCFeatureVersion());
      return -1;
  }

  /*
   * Perform runtime check of endianness and check it matches the one set by
   * the headers (npy_endian.h) as a safeguard
   */
  st = PyArray_GetEndianness();
  if (st == NPY_CPU_UNKNOWN_ENDIAN) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as unknown endian");
      return -1;
  }
#if NPY_BYTE_ORDER == NPY_BIG_ENDIAN
  if (st != NPY_CPU_BIG) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
             "big endian, but detected different endianness at runtime");
      return -1;
  }
#elif NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
  if (st != NPY_CPU_LITTLE) {
      PyErr_Format(PyExc_RuntimeError, "FATAL: module compiled as "\
             "little endian, but detected different endianness at runtime");
      return -1;
  }
#endif

  return 0;
}

#define import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return NULL; } }

#define import_array1(ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return ret; } }

#define import_array2(msg, ret) {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, msg); return ret; } }

// HPy
#define himport_array(ctx) {if (_import_array() < 0) {PyErr_Print(); HPyErr_SetString(ctx, ctx->h_ImportError, "numpy.core.multiarray failed to import"); return -1; } }

#define himport_array1(ctx, ret) {if (_import_array() < 0) {PyErr_Print(); HPyErr_SetString(ctx, ctx->h_ImportError, "numpy.core.multiarray failed to import"); return ret; } }

#define himport_array2(ctx, msg, ret) {if (_import_array() < 0) {PyErr_Print(); HPyErr_SetString(ctx, ctx->h_ImportError, msg); return ret; } }

#define hpy_import_array himport_array
#define hpy_import_array1 himport_array1
#define hpy_import_array2 himport_array2

#endif

#endif
"""


c_template = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
%s
};

static void init_array_api() {
%s
}

void *HPyArray_API[] = {
%s
};

static void init_hpy_array_api() {
%s
}
"""

c_api_header = """
===========
NumPy C-API
===========
"""

hpy_api_header = """
===========
NumPy HPy-API
===========
"""

def generate_api(output_dir, force=False):
    basename = 'multiarray_api'

    h_file = os.path.join(output_dir, '__%s.h' % basename)
    c_file = os.path.join(output_dir, '__%s.c' % basename)
    d_file = os.path.join(output_dir, '%s.txt' % basename)
    targets = (h_file, c_file, d_file)

    sources = numpy_api.multiarray_api

    if (not force and not genapi.should_rebuild(targets, [numpy_api.__file__, __file__])):
        return targets
    else:
        m, e, i, a, n = do_generate_api(sources)
        hpy_sources = numpy_api.hpy_multiarray_api
        print("*" * 50)
        print("Number of Multiarray APIs ported to HPy: %d / %d" % (len(hpy_sources[3]), len(sources[3])))
        print("*" * 50)
        hm, he, hi, ha, hn = do_generate_api(hpy_sources, 'HPyArray_API', 'HPY_NUMPY_API', 'HPyGlobal', hpy=True)
        # check for duplicates
        write_sources(targets, m, e, i, a, n, hm, he, hi, ha, hn)

    return targets

def do_generate_api(sources, 
                        api_name='PyArray_API', 
                        tagname='NUMPY_API',
                        py_type='PyTypeObject',
                        hpy=False):

    global_vars = sources[0]
    scalar_bool_values = sources[1]
    types_api = sources[2]
    multiarray_funcs = sources[3]

    multiarray_api = sources[:]

    module_list = []
    extension_list = []
    init_list = []
    assign_list = []

    # Check multiarray api indexes
    multiarray_api_index = genapi.merge_api_dicts(multiarray_api)
    genapi.check_api_dict(multiarray_api_index, hpy)

    numpyapi_list = genapi.get_api_functions(tagname,
                                             multiarray_funcs)

    # Create dict name -> *Api instance
    multiarray_api_dict = {}
    for f in numpyapi_list:
        name = f.name
        index = multiarray_funcs[name][0]
        annotations = multiarray_funcs[name][1:]
        multiarray_api_dict[f.name] = FunctionApi(f.name, index, annotations,
                                                  f.return_type,
                                                  f.args, api_name, hpy=hpy)

    for name, val in global_vars.items():
        index, type = val
        multiarray_api_dict[name] = GlobalVarApi(name, index, type, api_name, hpy=hpy)

    for name, val in scalar_bool_values.items():
        index = val[0]
        multiarray_api_dict[name] = BoolValuesApi(name, index, api_name, hpy=hpy)

    for name, val in types_api.items():
        index = val[0]
        internal_type = None if len(val) <= 1 else val[1]
        dynamic_init = None if len(val) <= 2 else val[2]
        multiarray_api_dict[name] = TypeApi(
            name, index, py_type, api_name, internal_type, dynamic_init, hpy=hpy)

    if len(multiarray_api_dict) != len(multiarray_api_index):
        keys_dict = set(multiarray_api_dict.keys())
        keys_index = set(multiarray_api_index.keys())
        api_type = '' if not hpy else 'HPy '
        raise AssertionError(
            "Multiarray {}API size mismatch - "
            "index has extra keys {}, dict has extra keys {}"
            .format(api_type, keys_index - keys_dict, keys_dict - keys_index)
        )

    extension_list = []
    cur_index = 0
    for name, index in genapi.order_dict(multiarray_api_index):
        api_item = multiarray_api_dict[name]
        extension_list.append(api_item.define_from_array_api_string())
        if cur_index != index[0]: # fill gap in the array (remove once HPy api is complete)
            for i in range(cur_index, index[0]):
                init_list.append("        NULL")
        init_list.append(api_item.array_api_define())
        cur_index = index[0] + 1
        assignment = api_item.array_api_assign()
        if assignment:
            assign_list.append(assignment)
        module_list.append(api_item.internal_define())
    
    return module_list, extension_list, init_list, assign_list, numpyapi_list


def write_sources(targets, 
                    module_list, 
                    extension_list, 
                    init_list, 
                    assign_list, 
                    numpyapi_list,
                    hpy_module_list=[],
                    hpy_extension_list=[],
                    hpy_init_list=[],
                    hpy_assign_list=[],
                    hpy_numpyapi_list=[]):
    header_file = targets[0]
    c_file = targets[1]
    doc_file = targets[2]
    # Write to header
    s = h_template % (
        '\n'.join(module_list),
        '\n'.join(hpy_module_list),
        '\n'.join(extension_list),
        '\n'.join(hpy_extension_list),
    )
    genapi.write_file(header_file, s)

    # Write to c-code
    s = c_template % (
        ',\n'.join(init_list),
        '\n'.join(assign_list),
        ',\n'.join(hpy_init_list),
        '\n'.join(hpy_assign_list),
    )
    genapi.write_file(c_file, s)

    # write to documentation
    s = c_api_header
    for func in numpyapi_list:
        s += func.to_ReST()
        s += '\n\n'
    if hpy_numpyapi_list:
        s += '\n\n'
        s += hpy_api_header
        for func in hpy_numpyapi_list:
            s += func.to_ReST()
            s += '\n\n'
    genapi.write_file(doc_file, s)

    return targets
