# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/hobie/Documents/Comp Vision/Lab 2"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/hobie/Documents/Comp Vision/Lab 2"

# Include any dependencies generated for this target.
include CMakeFiles/Correct.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Correct.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Correct.dir/flags.make

CMakeFiles/Correct.dir/correct.cpp.o: CMakeFiles/Correct.dir/flags.make
CMakeFiles/Correct.dir/correct.cpp.o: correct.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/hobie/Documents/Comp Vision/Lab 2/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Correct.dir/correct.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Correct.dir/correct.cpp.o -c "/home/hobie/Documents/Comp Vision/Lab 2/correct.cpp"

CMakeFiles/Correct.dir/correct.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Correct.dir/correct.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/hobie/Documents/Comp Vision/Lab 2/correct.cpp" > CMakeFiles/Correct.dir/correct.cpp.i

CMakeFiles/Correct.dir/correct.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Correct.dir/correct.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/hobie/Documents/Comp Vision/Lab 2/correct.cpp" -o CMakeFiles/Correct.dir/correct.cpp.s

CMakeFiles/Correct.dir/correct.cpp.o.requires:

.PHONY : CMakeFiles/Correct.dir/correct.cpp.o.requires

CMakeFiles/Correct.dir/correct.cpp.o.provides: CMakeFiles/Correct.dir/correct.cpp.o.requires
	$(MAKE) -f CMakeFiles/Correct.dir/build.make CMakeFiles/Correct.dir/correct.cpp.o.provides.build
.PHONY : CMakeFiles/Correct.dir/correct.cpp.o.provides

CMakeFiles/Correct.dir/correct.cpp.o.provides.build: CMakeFiles/Correct.dir/correct.cpp.o


# Object files for target Correct
Correct_OBJECTS = \
"CMakeFiles/Correct.dir/correct.cpp.o"

# External object files for target Correct
Correct_EXTERNAL_OBJECTS =

Correct: CMakeFiles/Correct.dir/correct.cpp.o
Correct: CMakeFiles/Correct.dir/build.make
Correct: /usr/local/lib/libopencv_dnn.so.4.1.1
Correct: /usr/local/lib/libopencv_gapi.so.4.1.1
Correct: /usr/local/lib/libopencv_highgui.so.4.1.1
Correct: /usr/local/lib/libopencv_ml.so.4.1.1
Correct: /usr/local/lib/libopencv_objdetect.so.4.1.1
Correct: /usr/local/lib/libopencv_photo.so.4.1.1
Correct: /usr/local/lib/libopencv_stitching.so.4.1.1
Correct: /usr/local/lib/libopencv_video.so.4.1.1
Correct: /usr/local/lib/libopencv_videoio.so.4.1.1
Correct: /usr/local/lib/libopencv_imgcodecs.so.4.1.1
Correct: /usr/local/lib/libopencv_calib3d.so.4.1.1
Correct: /usr/local/lib/libopencv_features2d.so.4.1.1
Correct: /usr/local/lib/libopencv_flann.so.4.1.1
Correct: /usr/local/lib/libopencv_imgproc.so.4.1.1
Correct: /usr/local/lib/libopencv_core.so.4.1.1
Correct: CMakeFiles/Correct.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/hobie/Documents/Comp Vision/Lab 2/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Correct"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Correct.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Correct.dir/build: Correct

.PHONY : CMakeFiles/Correct.dir/build

CMakeFiles/Correct.dir/requires: CMakeFiles/Correct.dir/correct.cpp.o.requires

.PHONY : CMakeFiles/Correct.dir/requires

CMakeFiles/Correct.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Correct.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Correct.dir/clean

CMakeFiles/Correct.dir/depend:
	cd "/home/hobie/Documents/Comp Vision/Lab 2" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/hobie/Documents/Comp Vision/Lab 2" "/home/hobie/Documents/Comp Vision/Lab 2" "/home/hobie/Documents/Comp Vision/Lab 2" "/home/hobie/Documents/Comp Vision/Lab 2" "/home/hobie/Documents/Comp Vision/Lab 2/CMakeFiles/Correct.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Correct.dir/depend
