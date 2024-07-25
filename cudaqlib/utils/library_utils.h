/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

namespace cudaqlib::__internal__ {

struct CUDAQLibraryData {
  std::string path;
};

#if defined(__APPLE__) && defined(__MACH__)
inline static void getCUDAQLibraryPath(CUDAQLibraryData *data) {
  auto nLibs = _dyld_image_count();
  for (uint32_t i = 0; i < nLibs; i++) {
    auto ptr = _dyld_get_image_name(i);
    std::string libName(ptr);
    if (libName.find("cudaq-operators") != std::string::npos) {
      auto casted = static_cast<CUDAQLibraryData *>(data);
      casted->path = std::string(ptr);
    }
  }
}
#else
inline static int getCUDAQLibraryPath(struct dl_phdr_info *info, size_t size,
                                      void *data) {
  std::string libraryName(info->dlpi_name);
  if (libraryName.find("cudaq-operators") != std::string::npos) {
    auto casted = static_cast<CUDAQLibraryData *>(data);
    casted->path = std::string(info->dlpi_name);
  }
  return 0;
}
#endif

inline static std::string getCUDAQLibraryPath() {
  __internal__::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  getCUDAQLibraryPath(&data);
#else
  dl_iterate_phdr(__internal__::getCUDAQLibraryPath, &data);
#endif
  return data.path;
}

} // namespace cudaqlib::__internal__
