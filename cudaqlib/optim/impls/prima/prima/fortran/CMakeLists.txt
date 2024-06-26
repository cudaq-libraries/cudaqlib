
add_library (primaf
  common/linalg.f90
  common/pintrf.f90
  common/string.f90
  common/evaluate.f90
  common/preproc.f90
  common/univar.f90
  common/powalg.f90
  common/history.f90
  common/xinbd.f90
  common/message.f90
  common/fprint.f90
  common/ratio.f90
  common/debug.F90
  common/consts.F90
  common/checkexit.f90
  common/inf.F90
  common/redrho.f90
  common/memory.F90
  common/selectx.f90
  common/huge.F90
  common/infnan.F90
  common/shiftbase.f90
  common/infos.f90
  cobyla/cobyla.f90
  cobyla/cobylb.f90
  cobyla/geometry.f90
  cobyla/initialize.f90
  cobyla/trustregion.f90
  cobyla/update.f90
  bobyqa/bobyqa.f90
  bobyqa/bobyqb.f90
  bobyqa/geometry.f90
  bobyqa/initialize.f90
  bobyqa/trustregion.f90
  bobyqa/rescue.f90
  bobyqa/update.f90
  lincoa/update.f90
  lincoa/initialize.f90
  lincoa/getact.f90
  lincoa/trustregion.f90
  lincoa/geometry.f90
  lincoa/lincob.f90
  lincoa/lincoa.f90
  newuoa/initialize.f90
  newuoa/trustregion.f90
  newuoa/geometry.f90
  newuoa/update.f90
  newuoa/newuob.f90
  newuoa/newuoa.f90
  uobyqa/initialize.f90
  uobyqa/update.f90
  uobyqa/geometry.f90
  uobyqa/trustregion.f90
  uobyqa/uobyqb.f90
  uobyqa/uobyqa.f90
)

set (PRIMA_REAL_PRECISION "64" CACHE STRING "real precision")
set (PRIMA_INTEGER_KIND "0" CACHE STRING "integer kind")
target_compile_definitions (primaf PUBLIC "PRIMA_REAL_PRECISION=${PRIMA_REAL_PRECISION}")
target_compile_definitions (primaf PUBLIC "PRIMA_INTEGER_KIND=${PRIMA_INTEGER_KIND}")
set_target_properties(primaf PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY mod)
target_include_directories (primaf PRIVATE common)
target_include_directories (primaf PUBLIC
  $<INSTALL_INTERFACE:include/prima/mod>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>)

# because of calcfc_internal nested subroutine in cobylb.f90
if (HAVE_WARN_EXECSTACK)
  target_link_options (primaf PUBLIC "-Wl,--no-warn-execstack")
endif ()

# On Windows, when building shared libs, visible functions must be explicitly listed.
# Unlike C there are no standard keywords to do this in Fortran but only compiler-specific pragmas:
# !GCC$ attributes dllexport :: bobyqa_c
# !DEC$ attributes dllexport :: bobyqa_c
# the downside is that gfortran will raise -Wattributes warnings, and Intel compilers will complain
# that the directives are not standard conforming. See https://github.com/libprima/prima/issues/70.
# Another option is to use .def files referencing exported symbols:
# https://learn.microsoft.com/en-us/cpp/build/exporting-from-a-dll-using-def-files?view=msvc-170
# Mangling is different between GNU and Intel/LLVM compiler families so we use a dedicated file for each.
# Symbol names can be seen in object files with the dumpbin tool (or objdump) to write a new .def:
# dumpbin /symbols cobyla.f90.obj
# 017 00000000 UNDEF  no_type       External     | COBYLA_MOD_mp_COBYLA
# x86_64-w64-mingw32-objdump -t cobyla.f90.obj
# [  4](sec  1)(fl 0x00)(ty   20)(scl   2) (nx 0) 0x000000000000218b __cobyla_mod_MOD_cobyla
if (WIN32 AND BUILD_SHARED_LIBS)
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/primaf-${CMAKE_Fortran_COMPILER_ID}.def)
    target_sources (primaf PRIVATE primaf-${CMAKE_Fortran_COMPILER_ID}.def)
  else ()
    message (STATUS "Assuming Intel symbol mangling")
    target_sources (primaf PRIVATE primaf-Intel.def)
  endif ()
endif ()

if (WIN32)
  set_target_properties(primaf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endif()

install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/prima)

macro (prima_add_f_test name)
  add_executable (example_${name}_fortran_exe EXCLUDE_FROM_ALL examples/${name}/${name}_example.f90)
  target_link_libraries (example_${name}_fortran_exe PRIVATE primaf)
  target_include_directories (example_${name}_fortran_exe PRIVATE ${CMAKE_BINARY_DIR}/fortran)
  set_target_properties(example_${name}_fortran_exe PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/examples/${name}/mod)
  if (WIN32)
    set_target_properties(example_${name}_fortran_exe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  endif()
  if (PRIMA_ENABLE_EXAMPLES)
    set_target_properties (example_${name}_fortran_exe PROPERTIES EXCLUDE_FROM_ALL FALSE)
  endif ()
  add_test (NAME example_${name}_fortran COMMAND example_${name}_fortran_exe)
  add_dependencies(examples example_${name}_fortran_exe)
endmacro ()

if (NOT PRIMA_ENABLE_TESTS)
  return()
endif()

prima_add_f_test (cobyla)
prima_add_f_test (bobyqa)
prima_add_f_test (newuoa)
prima_add_f_test (uobyqa)
prima_add_f_test (lincoa)
