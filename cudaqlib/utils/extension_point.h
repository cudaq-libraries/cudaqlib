/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <functional>
#include <memory>
#include <unordered_map>

namespace cudaq {
template <typename T, typename... CtorArgs>
class extension_point {
  using CreatorFunction = std::function<std::unique_ptr<T>(CtorArgs &&...)>;

protected:
  static std::unordered_map<std::string, CreatorFunction> &get_registry() {
    static std::unordered_map<std::string, CreatorFunction> registry;
    return registry;
  }

public:
  static std::unique_ptr<T> get(const std::string &name, CtorArgs &&...args) {
    auto &registry = get_registry();
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error("Cannot find extension with name = " + name);

    return iter->second(std::forward<CtorArgs>(args)...);
  }

  static std::vector<std::string> get_registered() {
    std::vector<std::string> names;
    auto &registry = get_registry();
    for (auto &[k, v] : registry)
      names.push_back(k);
    return names;
  }

  static bool is_registered(const std::string &name) {
    auto &registry = get_registry();
    return registry.find(name) != registry.end();
  }
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define CUDAQ_DEFINE_EXTENSION_IMPL(TYPE)                                      \
  namespace details {                                                          \
  template <class T>                                                           \
  class CONCAT(TYPE, _impl) : public TYPE {                                    \
  public:                                                                      \
    using TYPE::TYPE;                                                          \
    virtual ~CONCAT(TYPE, _impl)() {                                           \
      if (!registered_)                                                        \
        printf("this type was not registered\n");                              \
    }                                                                          \
                                                                               \
  protected:                                                                   \
    static inline bool register_type() {                                       \
      auto &registry = get_registry();                                         \
      registry[T::class_identifier] = T::create;                               \
      return true;                                                             \
    }                                                                          \
    static const bool registered_;                                             \
  };                                                                           \
  template <class T>                                                           \
  const bool CONCAT(TYPE, _impl)<T>::registered_ =                             \
      CONCAT(TYPE, _impl)<T>::register_type();                                 \
  } // namespace details

} // namespace cudaq