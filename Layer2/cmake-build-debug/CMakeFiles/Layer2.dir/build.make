# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\CLion 2020.3.3\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\CLion 2020.3.3\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = F:\Clion_jetbrains\LRIris\Layer2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Layer2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Layer2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Layer2.dir/flags.make

CMakeFiles/Layer2.dir/main.cpp.obj: CMakeFiles/Layer2.dir/flags.make
CMakeFiles/Layer2.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Layer2.dir/main.cpp.obj"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Layer2.dir\main.cpp.obj -c F:\Clion_jetbrains\LRIris\Layer2\main.cpp

CMakeFiles/Layer2.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Layer2.dir/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E F:\Clion_jetbrains\LRIris\Layer2\main.cpp > CMakeFiles\Layer2.dir\main.cpp.i

CMakeFiles/Layer2.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Layer2.dir/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S F:\Clion_jetbrains\LRIris\Layer2\main.cpp -o CMakeFiles\Layer2.dir\main.cpp.s

# Object files for target Layer2
Layer2_OBJECTS = \
"CMakeFiles/Layer2.dir/main.cpp.obj"

# External object files for target Layer2
Layer2_EXTERNAL_OBJECTS =

Layer2.exe: CMakeFiles/Layer2.dir/main.cpp.obj
Layer2.exe: CMakeFiles/Layer2.dir/build.make
Layer2.exe: CMakeFiles/Layer2.dir/linklibs.rsp
Layer2.exe: CMakeFiles/Layer2.dir/objects1.rsp
Layer2.exe: CMakeFiles/Layer2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Layer2.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Layer2.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Layer2.dir/build: Layer2.exe

.PHONY : CMakeFiles/Layer2.dir/build

CMakeFiles/Layer2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Layer2.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Layer2.dir/clean

CMakeFiles/Layer2.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" F:\Clion_jetbrains\LRIris\Layer2 F:\Clion_jetbrains\LRIris\Layer2 F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug F:\Clion_jetbrains\LRIris\Layer2\cmake-build-debug\CMakeFiles\Layer2.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Layer2.dir/depend
