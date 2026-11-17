#include "python_bindings.h"
#include "Graph.h"
#include "Fusion.h"
#include "GradMode.h"
#include <sstream>
#include <vector>

using namespace tensorplay::stax;


// ---------------------------------------------------------------------------
// Compiled-call trampoline (steady-state METH_FASTCALL entry).
//
// Uses the same capsule-carried
// PyCFunction that absorbs the per-call Python dispatch layers
// (api.optimized key memo -> lowering.__call__ route/bind memos -> execute)
// for the exact steady state -- same tensor objects, unchanged versions.
//
// Fast path:  arity ok, all args are the installed tensor exact-type, every
//   (id, version) matches the stored fingerprint, and no input requires grad
//   when this lowering owns an autograd plan.  Inference tensors carry no
//   version counter, so their fingerprint entry is the identity alone -- one
//   probe instead of a version read that would throw and clear.  Then the
//   cached input vector (args + attribute targets + constants) is refreshed
//   in place and handed straight to the native Graph.execute callable --
//   zero Python frames.  When the installed kernel exposes a direct entry
//   (``direct``), the steady state skips the graph callable entirely:
//   pointers are read from the C++ value holders, the kernel allocates its
//   pinned output in C, and the result comes back pre-wrapped.
// Divert:     anything else vectorcalls the Python lowering, which re-resolves
//   binding/route through its own memos.  The fingerprint is deliberately
//   left stale: a caller whose state changed at all already diverted once,
//   and an unchanged re-call over a stale entry cannot be fast by definition,
//   so the Python side rebinds through tpx_fingerprint_store after it has
//   fully resolved the slow call.
//
// In-place mutation bumps _version (ConvKernels guard contract), so a stale
// fast hit is impossible by construction; requires_grad rides the fingerprint
// and is additionally checked whenever a gradient plan exists.  Guard reads
// go through non-throwing C++ helpers (tpx_tensor_version /
// tpx_tensor_requires_grad) instead of Python attribute lookups.
// ---------------------------------------------------------------------------

#include "CPythonBridge.h"

using namespace tensorplay::python_c;

namespace {

// Sentinel version slot for inference tensors: they carry no version counter
// and are immutable, so the identity alone keys the entry.  The value is
// unreachable for a real counter (unsigned, always non-negative).
constexpr long long kFpNoVersion = -1;

// ---------------------------------------------------------------------------
// Captured-state certificate.
//
// A recorded region may have baked values it read from outside its call
// arguments: module attributes, the globals and closure cells its Python code
// loads, and the module tree itself -- identities, training flags and child
// bindings.  Replaying the recording is valid exactly while every one of those
// still holds the object it held when the region was recorded (and, for a
// tensor, the same version).
//
// The Python side works out *where* those places are, once; re-reading them is
// then one pass in C.  A site is certified by identity alone, which is the
// condition under which the recorded value cannot have moved, so a certificate
// that passes means the Python side would have read the same answer.
// ---------------------------------------------------------------------------
enum {
    kStateAttribute = 0,
    kStateGlobal = 1,
    kStateClosure = 2,
};

//: A site whose recorded value is not a versioned tensor.
constexpr long long kStateNotTensor = -2;

struct StateSite {
    int kind;
    PyObject* owner;    // strong: module holding the attribute, or the function
    PyObject* name;     // strong: attribute / global name; null for a closure
    Py_ssize_t index;   // closure cell index
    PyObject* value;    // strong: the object read at record time; null = nothing
    long long version;  // version counter when ``value`` is a tensor
};

struct StateModule {
    PyObject* module;               // strong
    PyObject* training;             // strong: the flag read at record time
    PyObject* children;             // strong: the module's child table
    std::vector<PyObject*> names;   // strong, in table order
    std::vector<PyObject*> values;  // strong, matching ``names``
};

struct StatePlan {
    std::vector<StateSite> sites;
    std::vector<StateModule> modules;
};

struct CallTrampolineState {
    PyObject* lowering;      // borrowed: owned by the compiled wrapper
    PyObject* fallback;      // borrowed: outer optimized wrapper, if present
    PyObject* exec_fn;       // strong: native Graph.execute bound method
    PyObject* tail;          // strong list: attribute targets + constants
    PyObject* head;          // strong list: fixed leading arguments, or null
    PyObject* tensor_type;   // strong: tensorplay.Tensor (exact-type check)
    Py_ssize_t nargs_expected;
    int output_count;
    bool gradient_plan;
    // Direct kernel entry (pinned fused kernels): returns the wrapped
    // output PyObject* for a C array of data pointers.  Null keeps the
    // exec_fn path.
    PyObject* (*direct)(const void* const* ins);
    // Fingerprint: per-arg (id, version-or-sentinel, requires_grad views).
    // Single writer: the dispatcher-level store below, called once per fully
    // resolved Python call.  An empty id vector means "not stored", so a
    // fresh trampoline never fast-hits before its first resolution.
    std::vector<PyObject*> fp_ids;      // strong refs pin the identity
    std::vector<long long> fp_versions;  // kFpNoVersion marks an inference tensor
    std::vector<int> fp_rg;      // fast-path view: 0 unless a gradient plan exists
    std::vector<int> fp_rg_live;  // memo view: live requires_grad flag
    // Captured-state certificate and the ambient autograd flag it was taken
    // under.  Present only where the recorded region's validity depends on
    // state the fingerprint cannot see; -1 means "not compared".
    StatePlan* state_plan;
    int grad_enabled_expected;
};

bool capture_busy_probe();

void destroy_state_plan(StatePlan* plan);

// The two names a module entry is re-read through.  Interned once: they are
// the same strings on every call, so the read costs a table lookup.
PyObject* interned_name(PyObject** slot, const char* name) {
    if (*slot == nullptr) {
        *slot = PyUnicode_InternFromString(name);
        if (*slot == nullptr) PyErr_Clear();
    }
    return *slot;
}

PyObject* plan_training_name() {
    static PyObject* name = nullptr;
    return interned_name(&name, "training");
}

PyObject* plan_children_name() {
    static PyObject* name = nullptr;
    return interned_name(&name, "_modules");
}

CallTrampolineState* trampoline_state_from_callable(PyObject* callable) {
    if (callable == nullptr || !PyCFunction_Check(callable)) return nullptr;
    PyObject* capsule = PyCFunction_GET_SELF(callable);
    if (capsule == nullptr || !PyCapsule_CheckExact(capsule)) return nullptr;
    void* pointer = PyCapsule_GetPointer(capsule, nullptr);
    if (pointer == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    return static_cast<CallTrampolineState*>(pointer);
}

void destroy_trampoline_state(CallTrampolineState* st) {
    Py_XDECREF(st->exec_fn);
    Py_XDECREF(st->tail);
    Py_XDECREF(st->head);
    Py_XDECREF(st->tensor_type);
    for (PyObject* id : st->fp_ids) Py_XDECREF(id);
    destroy_state_plan(st->state_plan);
    delete st;
}

// Destroy everything a plan holds.  Every entry owns a strong reference, so
// dropping the plan without this would keep the module tree alive.
void destroy_state_plan(StatePlan* plan) {
    if (plan == nullptr) return;
    for (StateSite& site : plan->sites) {
        Py_XDECREF(site.owner);
        Py_XDECREF(site.name);
        Py_XDECREF(site.value);
    }
    for (StateModule& entry : plan->modules) {
        Py_XDECREF(entry.module);
        Py_XDECREF(entry.training);
        Py_XDECREF(entry.children);
        for (PyObject* name : entry.names) Py_XDECREF(name);
        for (PyObject* value : entry.values) Py_XDECREF(value);
    }
    delete plan;
}

// The value an attribute site currently holds.  ``owner.__dict__`` is read
// first and the type's own table second -- the two places a Python attribute
// read reaches a captured value from -- and a name that is in neither reads
// nothing.  A null result without an error means "reads nothing".
PyObject* state_attribute(PyObject* owner, PyObject* name) {
    if (owner == nullptr || name == nullptr) return nullptr;
    PyObject** dict = nullptr;
    PyObject* type_table = nullptr;
    if (PyType_Check(owner)) {
        // A class keeps its attributes in its own table, and ``owner.__dict__``
        // is that table; the second place a read can reach is the metaclass.
        auto* type = reinterpret_cast<PyTypeObject*>(owner);
        dict = &type->tp_dict;
        type_table = Py_TYPE(type)->tp_dict;
    } else {
        dict = _PyObject_GetDictPtr(owner);
        type_table = Py_TYPE(owner)->tp_dict;
    }
    if (dict != nullptr && *dict != nullptr) {
        PyObject* value = PyDict_GetItemWithError(*dict, name);
        if (value != nullptr) return value;
        if (PyErr_Occurred()) {
            PyErr_Clear();
            return nullptr;
        }
    }
    if (type_table == nullptr) return nullptr;
    PyObject* value = PyDict_GetItemWithError(type_table, name);
    if (value != nullptr) return value;
    if (PyErr_Occurred()) PyErr_Clear();
    return nullptr;
}

// The value a global or closure site currently holds, or null for "nothing".
PyObject* state_shared_value(const StateSite& site) {
    if (site.kind == kStateClosure) {
        PyObject* closure = PyFunction_GET_CLOSURE(site.owner);
        if (closure == nullptr || !PyTuple_Check(closure)) return nullptr;
        if (site.index < 0 || site.index >= PyTuple_GET_SIZE(closure)) {
            return nullptr;
        }
        PyObject* cell = PyTuple_GET_ITEM(closure, site.index);
        if (!PyCell_Check(cell)) return nullptr;
        return PyCell_GET(cell);
    }
    PyObject* globals = PyFunction_GET_GLOBALS(site.owner);
    if (globals == nullptr) return nullptr;
    PyObject* value = PyDict_GetItemWithError(globals, site.name);
    if (value != nullptr) return value;
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return nullptr;
    }
    // A missing global reads as nothing; the Python side spells it ``None``,
    // which is the object a plan records for that site.
    return Py_None;
}

// True when every site and every module binding still holds what the region
// was recorded against.  Identity is the test; a tensor additionally has to
// carry the version it had, since an in-place update leaves the object in
// place while the value it describes moves.
bool state_plan_holds(const StatePlan& plan) {
    for (const StateSite& site : plan.sites) {
        PyObject* value = site.kind == kStateAttribute
                              ? state_attribute(site.owner, site.name)
                              : state_shared_value(site);
        if (value != site.value) return false;
        if (value == nullptr || site.version == kStateNotTensor) continue;
        long long version = 0;
        if (tpx_tensor_guard_probe(value, &version) != 0) return false;
        if (version != site.version) return false;
    }
    for (const StateModule& entry : plan.modules) {
        if (state_attribute(entry.module, plan_training_name()) != entry.training) {
            return false;
        }
        PyObject* children = state_attribute(entry.module, plan_children_name());
        if (children != entry.children) return false;
        if (!PyDict_Check(children)) return false;
        if (PyDict_GET_SIZE(children) != (Py_ssize_t)entry.names.size()) {
            return false;
        }
        PyObject *name = nullptr;
        PyObject *child = nullptr;
        Py_ssize_t position = 0;
        Py_ssize_t cursor = 0;
        while (PyDict_Next(children, &cursor, &name, &child)) {
            if (name != entry.names[position]) return false;
            if (child != entry.values[position]) return false;
            ++position;
        }
    }
    return true;
}

// Read the fingerprint-relevant state of one argument.  Returns false when
// the argument cannot be fingerprinted.  ``version`` is the counter value,
// or kFpNoVersion for an inference tensor.  ``rg_live`` is the live
// requires_grad flag when ``want_rg`` is set (the gradient-plan fast path
// skips one guard probe per arg otherwise) and 0 when not read.
bool fp_probe(CallTrampolineState* st, PyObject* v, bool want_rg,
              long long& version, int& rg_fast, int& rg_live) {
    if (Py_TYPE(v) != reinterpret_cast<PyTypeObject*>(st->tensor_type)) {
        return false;
    }
    long long vv;
    int probe = tpx_tensor_guard_probe(v, &vv);
    if (probe < 0) return false;
    if (probe == 1) vv = kFpNoVersion;
    rg_live = 0;
    if (want_rg) {
        rg_live = tpx_tensor_requires_grad(v);
        if (rg_live < 0) return false;
    }
    rg_fast = st->gradient_plan ? rg_live : 0;
    version = vv;
    return true;
}

// True when ``args`` exactly matches the stored fingerprint: same arity,
// every slot the same tensor object with the same (possibly absent) version
// and an unchanged live requires_grad flag.  Read-only: this is the C
// certificate the Python memo layer consumes instead of rebuilding an
// (id, version) tuple per diverted call.
bool trampoline_fingerprint_matches(const CallTrampolineState* st,
                                    PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != static_cast<Py_ssize_t>(st->fp_ids.size())) return false;
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject* v = args[i];
        if (Py_TYPE(v) != reinterpret_cast<PyTypeObject*>(st->tensor_type)) {
            return false;
        }
        long long vv;
        int probe = tpx_tensor_guard_probe(v, &vv);
        if (probe < 0) return false;
        if (probe == 1) vv = kFpNoVersion;
        if (st->fp_ids[i] != v || st->fp_versions[i] != vv) return false;
        int rgi = tpx_tensor_requires_grad(v);
        if (rgi < 0 || rgi != st->fp_rg_live[i]) return false;
    }
    return true;
}

void trampoline_fingerprint_reset(CallTrampolineState* st) {
    for (PyObject* id : st->fp_ids) Py_XDECREF(id);
    st->fp_ids.clear();
    st->fp_versions.clear();
    st->fp_rg.clear();
    st->fp_rg_live.clear();
}

void trampoline_fingerprint_store(CallTrampolineState* st,
                                  PyObject* const* args, Py_ssize_t nargs) {
    std::vector<PyObject*> ids;
    std::vector<long long> versions;
    std::vector<int> rg_fast;
    std::vector<int> rg_live;
    ids.reserve(static_cast<size_t>(nargs));
    versions.reserve(static_cast<size_t>(nargs));
    rg_fast.reserve(static_cast<size_t>(nargs));
    rg_live.reserve(static_cast<size_t>(nargs));
    bool ok = nargs >= 0;
    for (Py_ssize_t i = 0; ok && i < nargs; ++i) {
        long long vv;
        int rgf, rgl;
        if (!fp_probe(st, args[i], true, vv, rgf, rgl)) {
            ok = false;
            break;
        }
        ids.push_back(args[i]);
        versions.push_back(vv);
        rg_fast.push_back(rgf);
        rg_live.push_back(rgl);
    }
    if (!ok) {
        // A call the fingerprint cannot describe (scalars, kwargs-only
        // positionals, exotic objects) must not leave a stale certificate
        // behind: the Python memo would then pair old slots with new
        // quick parts.  Clear instead.
        trampoline_fingerprint_reset(st);
        return;
    }
    for (PyObject* id : st->fp_ids) Py_XDECREF(id);
    st->fp_ids = std::move(ids);
    st->fp_versions = std::move(versions);
    st->fp_rg = std::move(rg_fast);
    st->fp_rg_live = std::move(rg_live);
    for (PyObject* id : st->fp_ids) Py_INCREF(id);
}

PyObject* call_trampoline(PyObject* self, PyObject* const* args,
                          Py_ssize_t nargs, PyObject* kwnames) {
    auto* st = static_cast<CallTrampolineState*>(PyCapsule_GetPointer(self, nullptr));
    if (!st) return nullptr;

    if (capture_busy_probe()) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }

    bool can_fast = (kwnames == nullptr && nargs == st->nargs_expected);
    if (can_fast) {
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            PyObject* v = args[i];
            long long vv;
            int rg_fast, rg_live;
            if (!fp_probe(st, v, st->gradient_plan, vv, rg_fast, rg_live) ||
                static_cast<size_t>(i) >= st->fp_ids.size() ||
                st->fp_ids[i] != v || st->fp_versions[i] != vv ||
                st->fp_rg[i] != rg_fast) {
                // Leave the stored fingerprint untouched: the diverted Python
                // resolution is the single writer and re-stores it after the
                // slow path has fully re-resolved, so slots always describe
                // the call the quick memo was built from.
                can_fast = false;
                break;
            }
            if (st->gradient_plan && rg_live) {
                // Autograd route decided by the Python lowering: any input
                // demanding grad diverts.
                can_fast = false;
                break;
            }
        }
    }

    if (!can_fast) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }

    // State the fingerprint cannot see: the ambient autograd switch the
    // recording was taken under, and every place the region read a value from
    // outside its arguments.  Either one moving means the recording no longer
    // describes this call.
    if (st->state_plan != nullptr &&
        !state_plan_holds(*st->state_plan)) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }
    if (st->grad_enabled_expected >= 0 &&
        tensorplay::GradMode::is_enabled() !=
            (st->grad_enabled_expected != 0)) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }

    if (st->direct != nullptr) {
        // Steady state with a pinned kernel: pointers straight from the
        // C++ value holders, kernel allocates and wraps its own output.
        const void* inline_ins[16];
        std::vector<const void*> dynamic_ins;
        const void** ins = inline_ins;
        if (nargs > static_cast<Py_ssize_t>(
                         sizeof(inline_ins) / sizeof(inline_ins[0]))) {
            dynamic_ins.resize(static_cast<size_t>(nargs));
            ins = dynamic_ins.data();
        }
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            ins[i] = tpx_py_tensor_cref(args[i]).unsafeGetTensorImpl()->data();
        }
        return st->direct(ins);
    }

    Py_ssize_t head_count = st->head != nullptr ? PyList_GET_SIZE(st->head) : 0;
    Py_ssize_t total = head_count + nargs + PyList_GET_SIZE(st->tail);
    PyObject* inputs = PyList_New(total);
    if (!inputs) return nullptr;
    for (Py_ssize_t i = 0; i < head_count; ++i) {
        PyObject* item = PyList_GET_ITEM(st->head, i);  // borrowed
        Py_INCREF(item);
        PyList_SET_ITEM(inputs, i, item);
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(args[i]);
        PyList_SET_ITEM(inputs, head_count + i, args[i]);
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(st->tail); ++i) {
        PyObject* item = PyList_GET_ITEM(st->tail, i);  // borrowed
        Py_INCREF(item);
        PyList_SET_ITEM(inputs, head_count + nargs + i, item);
    }
    PyObject* out = PyObject_CallOneArg(st->exec_fn, inputs);
    Py_DECREF(inputs);
    if (!out) return nullptr;
    if (st->output_count == 1 &&
        (PyList_CheckExact(out) || PyTuple_CheckExact(out)) &&
        PySequence_Fast_GET_SIZE(out) == 1) {
        // Graph.execute returns py::list; Python lowering unwraps single
        // outputs -- preserve that contract exactly.
        PyObject* single = PySequence_Fast_GET_ITEM(out, 0);  // borrowed
        Py_INCREF(single);
        Py_DECREF(out);
        return single;
    }
    if (PyList_CheckExact(out)) {
        PyObject* as_tuple = PyList_AsTuple(out);
        Py_DECREF(out);
        return as_tuple;  // multi-output: match Python's tuple contract
    }
    return out;
}

// Read one site of a plan description into ``site``.  The description states
// where the value comes from, and whether the site read anything at all.
bool build_state_site(PyObject* item, StateSite* site) {
    PyObject* parts = PySequence_Fast(item, "a state site is a sequence");
    if (parts == nullptr) return false;
    bool ok = PySequence_Fast_GET_SIZE(parts) == 5;
    if (ok) {
        PyObject* kind = PySequence_Fast_GET_ITEM(parts, 0);
        PyObject* owner = PySequence_Fast_GET_ITEM(parts, 1);
        PyObject* key = PySequence_Fast_GET_ITEM(parts, 2);
        PyObject* value = PySequence_Fast_GET_ITEM(parts, 3);
        PyObject* present = PySequence_Fast_GET_ITEM(parts, 4);
        long kind_of = PyLong_AsLong(kind);
        if (kind_of == -1 && PyErr_Occurred()) {
            ok = false;
        } else if (kind_of == kStateClosure) {
            Py_ssize_t index = PyLong_AsSsize_t(key);
            if (index == -1 && PyErr_Occurred()) {
                ok = false;
            } else {
                site->kind = kStateClosure;
                site->owner = Py_NewRef(owner);
                site->name = nullptr;
                site->index = index;
            }
        } else if (kind_of == kStateAttribute || kind_of == kStateGlobal) {
            site->kind = static_cast<int>(kind_of);
            site->owner = Py_NewRef(owner);
            site->name = Py_NewRef(key);
            site->index = 0;
        } else {
            PyErr_SetString(PyExc_ValueError, "unknown state site kind");
            ok = false;
        }
        if (ok) {
            int has_value = PyObject_IsTrue(present);
            if (has_value < 0) {
                ok = false;
            } else if (has_value == 0) {
                site->value = nullptr;
                site->version = kStateNotTensor;
            } else {
                site->value = Py_NewRef(value);
                site->version = kStateNotTensor;
                long long version = 0;
                // A versioned tensor is certified by its counter as well:
                // the object stays the same across an in-place update while
                // the value it describes does not.
                if (tpx_tensor_guard_probe(value, &version) == 0) {
                    site->version = version;
                }
            }
        }
    }
    Py_DECREF(parts);
    return ok;
}

// Read one module entry of a plan description: the module, the training flag
// it carried, and its child bindings in table order.
bool build_state_module(PyObject* item, StateModule* entry) {
    PyObject* parts = PySequence_Fast(item, "a state module is a sequence");
    if (parts == nullptr) return false;
    bool ok = PySequence_Fast_GET_SIZE(parts) == 3;
    if (ok) {
        PyObject* module = PySequence_Fast_GET_ITEM(parts, 0);
        PyObject* training = PySequence_Fast_GET_ITEM(parts, 1);
        PyObject* children = PySequence_Fast_GET_ITEM(parts, 2);
        PyObject* table = state_attribute(module, plan_children_name());
        if (table == nullptr || !PyDict_Check(table)) {
            PyErr_SetString(PyExc_TypeError, "a state module has no child table");
            ok = false;
        } else {
            entry->module = Py_NewRef(module);
            entry->training = Py_NewRef(training);
            entry->children = Py_NewRef(table);
            Py_ssize_t count = PySequence_Fast_GET_SIZE(children);
            if (count < 0) {
                ok = false;
            } else {
                entry->names.reserve(static_cast<size_t>(count));
                entry->values.reserve(static_cast<size_t>(count));
                for (Py_ssize_t i = 0; ok && i < count; ++i) {
                    PyObject* pair = PySequence_Fast_GET_ITEM(children, i);
                    PyObject* pair_parts = PySequence_Fast(pair, "a child binding is a sequence");
                    if (pair_parts == nullptr || PySequence_Fast_GET_SIZE(pair_parts) != 2) {
                        Py_XDECREF(pair_parts);
                        ok = false;
                        break;
                    }
                    entry->names.push_back(Py_NewRef(PySequence_Fast_GET_ITEM(pair_parts, 0)));
                    entry->values.push_back(Py_NewRef(PySequence_Fast_GET_ITEM(pair_parts, 1)));
                    Py_DECREF(pair_parts);
                }
                if (ok && (Py_ssize_t)entry->names.size() != PyDict_GET_SIZE(table)) {
                    PyErr_SetString(PyExc_ValueError,
                                    "a state module's child list does not match its table");
                    ok = false;
                }
            }
        }
    }
    Py_DECREF(parts);
    return ok;
}

// Build a plan from (modules, sites).  Every entry takes a strong reference,
// so the plan keeps the objects it certifies alive and their addresses are
// never reused underneath it.
StatePlan* build_state_plan(PyObject* description) {
    if (!PyTuple_Check(description) || PyTuple_GET_SIZE(description) != 2) {
        PyErr_SetString(PyExc_TypeError, "a state plan is (modules, sites)");
        return nullptr;
    }
    auto* plan = new StatePlan();
    auto fail = [&]() -> StatePlan* {
        destroy_state_plan(plan);
        return nullptr;
    };
    PyObject* modules = PySequence_Fast(PyTuple_GET_ITEM(description, 0),
                                        "a state plan's modules");
    if (modules == nullptr) return fail();
    PyObject* sites = PySequence_Fast(PyTuple_GET_ITEM(description, 1),
                                      "a state plan's sites");
    if (sites == nullptr) {
        Py_DECREF(modules);
        return fail();
    }
    Py_ssize_t module_count = PySequence_Fast_GET_SIZE(modules);
    Py_ssize_t site_count = PySequence_Fast_GET_SIZE(sites);
    plan->modules.reserve(static_cast<size_t>(module_count));
    plan->sites.reserve(static_cast<size_t>(site_count));
    bool ok = true;
    for (Py_ssize_t i = 0; ok && i < module_count; ++i) {
        StateModule entry{};
        if (!build_state_module(PySequence_Fast_GET_ITEM(modules, i), &entry)) {
            // Free what this entry took before dropping the plan: the plan
            // only owns entries that were pushed.
            Py_XDECREF(entry.module);
            Py_XDECREF(entry.training);
            Py_XDECREF(entry.children);
            for (PyObject* name : entry.names) Py_XDECREF(name);
            for (PyObject* value : entry.values) Py_XDECREF(value);
            ok = false;
            break;
        }
        plan->modules.push_back(std::move(entry));
    }
    for (Py_ssize_t i = 0; ok && i < site_count; ++i) {
        StateSite site{};
        site.owner = nullptr;
        site.name = nullptr;
        site.value = nullptr;
        site.version = kStateNotTensor;
        site.index = 0;
        if (!build_state_site(PySequence_Fast_GET_ITEM(sites, i), &site)) {
            Py_XDECREF(site.owner);
            Py_XDECREF(site.name);
            Py_XDECREF(site.value);
            ok = false;
            break;
        }
        plan->sites.push_back(site);
    }
    Py_DECREF(modules);
    Py_DECREF(sites);
    if (!ok) return fail();
    return plan;
}

PyMethodDef trampoline_def = {
    "call_trampoline", (PyCFunction)(void(*)(void))call_trampoline,
    METH_FASTCALL | METH_KEYWORDS, nullptr,
};

// ---------------------------------------------------------------------------
// Compiled wrapper dispatcher: the outermost entry a user actually calls.
//
// The Python wrapper (api.compile's ``optimized``) owns capture checks, the
// specialization cache and guard chains, but its steady state is: same
// objects, unchanged versions, no capture, no gates -- work the trampoline
// already certifies.  This type short-circuits that Python frame: one C
// vectorcall hands the call to the trampoline, which owns the fingerprint
// check and the slow-path diversion.
// Anything else (new tensors, version bumps, kwargs, capture active)
// vectorcalls the Python wrapper, which resolves everything and rebinds
// ``fast`` through the ``tpx_set_fast`` method below.
// ---------------------------------------------------------------------------

struct CallDispatcher {
    PyObject_HEAD
    vectorcallfunc vectorcall = nullptr;
    // The GC allocator hands out raw memory; pointer fields are initialized
    // by the installer and released by tp_clear/dealloc.
    PyObject* python_entry;   // strong: api.compile's optimized wrapper
    PyObject* fast;           // strong: trampoline PyCFunction (or null)
    PyObject* dict;           // optional instance attributes
    Py_ssize_t nargs;
};
PyObject* call_dispatcher_call(PyObject* self, PyObject* const* args,
                               size_t nargsf, PyObject* kwnames) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        return PyObject_Vectorcall(d->fast, args, nargsf, kwnames);
    }
    return PyObject_Vectorcall(d->python_entry, args, nargsf, kwnames);
}

PyObject* call_dispatcher_getattro(PyObject* self, PyObject* name) {
    // Type-level surface (tpx_set_fast) resolves normally; everything else
    // forwards to the Python wrapper -- attribute traffic is not
    // steady-state.
    if (PyObject* res = PyObject_GenericGetAttr(self, name)) {
        return res;
    }
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
        return nullptr;
    }
    PyErr_Clear();
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    return PyObject_GetAttr(d->python_entry, name);
}

PyObject* call_dispatcher_set_fast(PyObject* self, PyObject* const* args,
                                   Py_ssize_t nargs) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tpx_set_fast(fast, nargs)");
        return nullptr;
    }
    PyObject* old_fast = d->fast;
    d->fast = nullptr;
    if (old_fast != nullptr) {
        if (auto* old_state = trampoline_state_from_callable(old_fast)) {
            old_state->fallback = nullptr;
        }
        Py_DECREF(old_fast);
    }
    if (args[0] != Py_None) {
        d->fast = args[0];
        Py_INCREF(d->fast);
        d->nargs = PyNumber_AsSsize_t(args[1], PyExc_OverflowError);
        if (d->nargs == -1 && PyErr_Occurred()) {
            Py_CLEAR(d->fast);
            return nullptr;
        }
        if (auto* state = trampoline_state_from_callable(d->fast)) {
            state->fallback = d->python_entry;
        }
    }
    Py_RETURN_NONE;
}

// Read-only certificate for the Python memo layer: true when the currently
// bound trampoline's stored fingerprint describes exactly these arguments.
// The Python wrapper consumes this instead of rebuilding an (id, version)
// tuple per diverted call -- one C probe replaces a Python loop, and the
// single-writer store below keeps the certificate paired with the memo it
// validates.
PyObject* call_dispatcher_fingerprint_matches(PyObject* self,
                                              PyObject* const* args,
                                              Py_ssize_t nargs) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    bool matched = false;
    if (d->fast != nullptr) {
        if (auto* st = trampoline_state_from_callable(d->fast)) {
            matched = trampoline_fingerprint_matches(st, args, nargs);
        }
    }
    if (matched) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

// Single writer of the steady-state fingerprint: called once per fully
// resolved Python call with that call's positional arguments.  Arguments
// the fingerprint cannot describe clear the stored state instead, so a
// stored fingerprint is always paired with the memo built alongside it.
PyObject* call_dispatcher_fingerprint_store(PyObject* self,
                                            PyObject* const* args,
                                            Py_ssize_t nargs) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        if (auto* st = trampoline_state_from_callable(d->fast)) {
            trampoline_fingerprint_store(st, args, nargs);
        }
    }
    Py_RETURN_NONE;
}

// Invalidate the stored fingerprint (specialization switch to a lowering
// without a trampoline, keyword-bearing calls): no certificate may outlive
// the memo it was stored with.
PyObject* call_dispatcher_fingerprint_clear(PyObject* self,
                                            PyObject* const* /*args*/,
                                            Py_ssize_t /*nargs*/) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        if (auto* st = trampoline_state_from_callable(d->fast)) {
            trampoline_fingerprint_reset(st);
        }
    }
    Py_RETURN_NONE;
}

// Install the captured-state certificate on the currently bound trampoline,
// paired with the ambient autograd switch it was taken under.  The trampoline
// consults both before replaying; a plan of None drops the certificate, which
// makes every call divert to the Python wrapper.
PyObject* call_dispatcher_set_state_plan(PyObject* self,
                                         PyObject* const* args,
                                         Py_ssize_t nargs) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tpx_set_state_plan(plan, grad_enabled)");
        return nullptr;
    }
    if (d->fast == nullptr) Py_RETURN_NONE;
    auto* st = trampoline_state_from_callable(d->fast);
    if (st == nullptr) Py_RETURN_NONE;
    destroy_state_plan(st->state_plan);
    st->state_plan = nullptr;
    st->grad_enabled_expected = -1;
    if (args[0] != Py_None) {
        st->state_plan = build_state_plan(args[0]);
        if (st->state_plan == nullptr) return nullptr;
        int enabled = PyObject_IsTrue(args[1]);
        if (enabled < 0) return nullptr;
        st->grad_enabled_expected = enabled;
    }
    Py_RETURN_NONE;
}

int call_dispatcher_traverse(PyObject* self, visitproc visit, void* arg) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    Py_VISIT(d->python_entry);
    Py_VISIT(d->fast);
    Py_VISIT(d->dict);
    return 0;
}

int call_dispatcher_clear(PyObject* self) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        if (auto* state = trampoline_state_from_callable(d->fast)) {
            state->fallback = nullptr;
        }
    }
    Py_CLEAR(d->python_entry);
    Py_CLEAR(d->fast);
    Py_CLEAR(d->dict);
    return 0;
}

void call_dispatcher_dealloc(PyObject* self) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    PyObject_GC_UnTrack(self);
    call_dispatcher_clear(self);
    Py_TYPE(self)->tp_free(self);
}

// Exposes the capture-busy probe used by the dispatcher
// to decide between the C steady state and the Python wrapper.
bool capture_busy_probe() {
    CaptureState state = currentCaptureState();
    return state.disabled_depth > 0 || state.compile_depth > 0 ||
           state.exporting_depth > 0;
}

PyMethodDef dispatcher_methods[] = {
    {"tpx_set_fast", (PyCFunction)(void(*)(void))call_dispatcher_set_fast,
     METH_FASTCALL, nullptr},
    {"tpx_fingerprint_matches",
     (PyCFunction)(void(*)(void))call_dispatcher_fingerprint_matches,
     METH_FASTCALL, nullptr},
    {"tpx_fingerprint_store",
     (PyCFunction)(void(*)(void))call_dispatcher_fingerprint_store,
     METH_FASTCALL, nullptr},
    {"tpx_fingerprint_clear",
     (PyCFunction)(void(*)(void))call_dispatcher_fingerprint_clear,
     METH_FASTCALL, nullptr},
    {"tpx_set_state_plan",
     (PyCFunction)(void(*)(void))call_dispatcher_set_state_plan,
     METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject CallDispatcherType = []() {
    PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
    t.tp_name = "tensorplay._C._stax.CallDispatcher";
    t.tp_basicsize = sizeof(CallDispatcher);
    t.tp_dealloc = call_dispatcher_dealloc;
    t.tp_call = PyVectorcall_Call;
    t.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                 Py_TPFLAGS_HAVE_VECTORCALL;
    t.tp_doc = "C dispatch entry for a compiled TensorPlay wrapper";
    t.tp_traverse = call_dispatcher_traverse;
    t.tp_clear = call_dispatcher_clear;
    t.tp_getattro = call_dispatcher_getattro;
    t.tp_setattro = PyObject_GenericSetAttr;
    t.tp_methods = dispatcher_methods;
    t.tp_vectorcall_offset = offsetof(CallDispatcher, vectorcall);
    t.tp_dictoffset = offsetof(CallDispatcher, dict);
    return t;
}();

}  // namespace

void init_stax(py::module_& m) {
    py::module_ stax_m = m.def_submodule("_stax", "Stax Static Graph Optimization");

    if (PyType_Ready(&CallDispatcherType) < 0) {
        throw py::error_already_set();
    }

    py::class_<Graph>(stax_m, "Graph")
        .def(py::init<>())
        .def("print", &Graph::print)
        .def("create_node", &Graph::createNode, py::return_value_policy::reference, py::arg("op_type"), py::arg("name") = "")
        .def("add_input", &Graph::addInput, py::return_value_policy::reference)
        .def("register_output", &Graph::registerOutput)
        .def("execute", &Graph::execute, py::arg("inputs"))
        .def("fuse", [](Graph& self) {
            fuseGraph(self);
        })
        .def_property_readonly("nodes", [](const Graph& g) {
            std::vector<OpNode*> nodes;
            for(auto& n : g.nodes) nodes.push_back(n.get());
            return nodes;
        }, py::return_value_policy::reference)
        .def_property_readonly("inputs", [](const Graph& g) { return g.inputs; }, py::return_value_policy::reference)
        .def_property_readonly("outputs", [](const Graph& g) { return g.outputs; }, py::return_value_policy::reference);

    py::class_<OpNode>(stax_m, "OpNode")
        .def_property("op_type", [](const OpNode& n) { return n.op_type; }, [](OpNode& n, const std::string& k) { n.op_type = k; })
        .def_property_readonly("name", [](const OpNode& n) { return n.name; })
        .def_property_readonly("input_count", [](const OpNode& n) { return n.inputs.size(); })
        .def("add_input", &OpNode::addInput)
        .def("add_output", &OpNode::addOutput, py::return_value_policy::reference)
        .def_property_readonly("inputs", [](const OpNode& n) { return n.inputs; }, py::return_value_policy::reference)
        .def_property_readonly("outputs", [](const OpNode& n) { return n.outputs; }, py::return_value_policy::reference)
        .def("set_int_attr", [](OpNode& n, const std::string& key, int64_t val) { n.setAttr(key, val); })
        .def("set_float_attr", [](OpNode& n, const std::string& key, double val) { n.setAttr(key, val); })
        .def("set_str_attr", [](OpNode& n, const std::string& key, const std::string& val) { n.setAttr(key, val); })
        .def("set_ints_attr", [](OpNode& n, const std::string& key, const std::vector<int64_t>& val) { n.setAttr(key, val); })
        .def("set_floats_attr", [](OpNode& n, const std::string& key, const std::vector<double>& val) { n.setAttr(key, val); })
        .def("get_int_attr", [](OpNode& n, const std::string& key) { 
            return n.getAttr<int64_t>(key);
        })
        .def("get_float_attr", [](OpNode& n, const std::string& key) {
            return n.getAttr<double>(key);
        })
        .def("has_attr", [](OpNode& n, const std::string& key) { return n.attrs.count(key) > 0; });
    
    py::class_<ValueNode>(stax_m, "ValueNode")
        .def_readonly("id", &ValueNode::id)
        .def_property("shape", 
            [](const ValueNode& v) { return v.shape; },
            [](ValueNode& v, const std::vector<int64_t>& s) { v.shape = s; })
        .def_property("dtype", 
            [](const ValueNode& v) { return v.dtype; },
            [](ValueNode& v, const std::string& d) { v.dtype = d; })
        .def_property_readonly("use_count", [](const ValueNode& v) {
            return v.uses.size();
        });

            
    py::class_<IRBuilder>(stax_m, "IRBuilder")
        .def(py::init<Graph&>())
        .def("create_input", &IRBuilder::createInput, py::return_value_policy::reference, py::arg("shape"), py::arg("dtype")="float32")
        .def("create_op", &IRBuilder::createOp, py::return_value_policy::reference, 
             py::arg("op_type"), py::arg("inputs"), py::arg("out_shape")=std::vector<int64_t>{}, py::arg("name")="")
        .def("mark_output", &IRBuilder::markOutput);

    // Compiled-call trampoline installer: see the CallTrampolineState block
    // above for the fast-path/divert contract.  ``direct_addr`` is the
    // address of a kernel module's ``tp_direct`` entry (ctypes round-trips
    // it as an integer); zero keeps the Graph.execute path.
    stax_m.def(
        "install_call_trampoline",
        [](py::object lowering, py::object exec_fn, py::list tail,
           py::object tensor_type, Py_ssize_t nargs_expected,
           int output_count, bool gradient_plan,
           unsigned long long direct_addr, py::object head) -> py::object {
            auto* st = new CallTrampolineState();
            st->lowering = lowering.ptr();
            st->fallback = nullptr;
            st->exec_fn = exec_fn.ptr();
            st->tail = tail.ptr();
            st->head = nullptr;
            st->tensor_type = tensor_type.ptr();
            st->nargs_expected = nargs_expected;
            st->output_count = output_count;
            st->gradient_plan = gradient_plan;
            st->state_plan = nullptr;
            st->grad_enabled_expected = -1;
            st->direct = reinterpret_cast<PyObject* (*)(const void* const*)>(
                direct_addr);
            Py_INCREF(st->exec_fn);
            Py_INCREF(st->tail);
            Py_INCREF(st->tensor_type);
            // A fixed leading list is the arguments a region always reads
            // before the caller's own: the state its program was recorded
            // against.  The fingerprint covers the caller's arguments, so the
            // list is what a call needs to rebuild the full argument vector
            // without walking the module tree again.
            if (!head.is_none()) {
                st->head = head.ptr();
                Py_INCREF(st->head);
            }
            PyObject* cap = PyCapsule_New((void*)st, nullptr,
                                          [](PyObject* c) {
                auto* hh = (CallTrampolineState*)PyCapsule_GetPointer(c, nullptr);
                if (!hh) return;
                destroy_trampoline_state(hh);
            });
            if (!cap) {
                destroy_trampoline_state(st);
                throw py::error_already_set();
            }
            PyObject* fn = PyCFunction_New(&trampoline_def, cap);
            Py_DECREF(cap);  // fn holds the self reference
            if (!fn) throw py::error_already_set();
            return py::reinterpret_steal<py::object>(fn);
        },
        py::arg("lowering"), py::arg("exec_fn"), py::arg("tail"),
        py::arg("tensor_type"), py::arg("nargs_expected"),
        py::arg("output_count"), py::arg("gradient_plan"),
        py::arg("direct_addr") = 0ULL, py::arg("head") = py::none());

    stax_m.def(
        "capture_state_enter",
        [](bool compiling, bool exporting, bool disabled) {
            enterCaptureState(compiling, exporting, disabled);
        },
        py::arg("compiling") = false,
        py::arg("exporting") = false,
        py::arg("disabled") = false);
    stax_m.def(
        "capture_state_exit",
        [](bool compiling, bool exporting, bool disabled) {
            exitCaptureState(compiling, exporting, disabled);
        },
        py::arg("compiling") = false,
        py::arg("exporting") = false,
        py::arg("disabled") = false);
    stax_m.def("capture_state", []() {
        CaptureState state = currentCaptureState();
        return py::make_tuple(
            state.compile_depth,
            state.disabled_depth,
            state.exporting_depth);
    });
    // Boolean probe for the compiled wrapper dispatch fast path: nonzero
    // when capture is disabled or a compile/export pass is running, i.e.
    // when the wrapper must fall back to the eager function instead of the
    // C steady-state launch.  ContextVar reads stay in the slow path.
    stax_m.def("capture_busy", []() {
        CaptureState state = currentCaptureState();
        return state.disabled_depth > 0 || state.compile_depth > 0 ||
               state.exporting_depth > 0;
    });
    // Compiled wrapper dispatcher installer: see the CallDispatcher block.
    // The dispatcher forwards attribute traffic to ``python_entry`` and
    // rebinds the steady-state target through ``tpx_set_fast``.
    stax_m.def(
        "make_call_dispatcher",
        [](py::object python_entry) -> py::object {
            auto* d = PyObject_GC_New(CallDispatcher, &CallDispatcherType);
            if (!d) throw py::error_already_set();
            d->vectorcall = call_dispatcher_call;
            d->python_entry = python_entry.ptr();
            Py_INCREF(d->python_entry);
            d->fast = nullptr;
            d->dict = nullptr;
            d->nargs = 0;
            PyObject_GC_Track(d);
            return py::reinterpret_steal<py::object>(
                reinterpret_cast<PyObject*>(d));
        },
        py::arg("python_entry"));

}
