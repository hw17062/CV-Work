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
CMAKE_SOURCE_DIR = "/home/hobie/Documents/Comp Vision/hobie"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/hobie/Documents/Comp Vision/hobie"

# Include any dependencies generated for this target.
include CMakeFiles/Board.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Board.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Board.dir/flags.make

CMakeFiles/Board.dir/board.cpp.o: CMakeFiles/Board.dir/flags.make
CMakeFiles/Board.dir/board.cpp.o: board.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/hobie/Documents/Comp Vision/hobie/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Board.dir/board.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Board.dir/board.cpp.o -c "/home/hobie/Documents/Comp Vision/hobie/board.cpp"

CMakeFiles/Board.dir/board.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Board.dir/board.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/hobie/Documents/Comp Vision/hobie/board.cpp" > CMakeFiles/Board.dir/board.cpp.i

CMakeFiles/Board.dir/board.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Board.dir/board.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/hobie/Documents/Comp Vision/hobie/board.cpp" -o CMakeFiles/Board.dir/board.cpp.s

CMakeFiles/Board.dir/board.cpp.o.requires:

.PHONY : CMakeFiles/Board.dir/board.cpp.o.requires

CMakeFiles/Board.dir/board.cpp.o.provides: CMakeFiles/Board.dir/board.cpp.o.requires
	$(MAKE) -f CMakeFiles/Board.dir/build.make CMakeFiles/Board.dir/board.cpp.o.provides.build
.PHONY : CMakeFiles/Board.dir/board.cpp.o.provides

CMakeFiles/Board.dir/board.cpp.o.provides.build: CMakeFiles/Board.dir/board.cpp.o


# Object files for target Board
Board_OBJECTS = \
"CMakeFiles/Board.dir/board.cpp.o"

# External object files for target Board
Board_EXTERNAL_OBJECTS =

Board: CMakeFiles/Board.dir/board.cpp.o
Board: CMakeFiles/Board.dir/build.make
Board: /usr/local/lib/libopencv_dnn.so.4.1.1
Board: /usr/local/lib/libopencv_gapi.so.4.1.1
Board: /usr/local/lib/libopencv_highgui.so.4.1.1
Board: /usr/local/lib/libopencv_ml.so.4.1.1
Board: /usr/local/lib/libopencv_objdetect.so.4.1.1
Board: /usr/local/lib/libopencv_photo.so.4.1.1
Board: /usr/local/lib/libopencv_stitching.so.4.1.1
Board: /usr/local/lib/libopencv_video.so.4.1.1
Board: /usr/local/lib/libopencv_videoio.so.4.1.1
Board: /usr/local/lib/libopencv_imgcodecs.so.4.1.1
Board: /usr/local/lib/libopencv_calib3d.so.4.1.1
Board: /usr/local/lib/libopencv_features2d.so.4.1.1
Board: /usr/local/lib/libopencv_flann.so.4.1.1
Board: /usr/local/lib/libopencv_imgproc.so.4.1.1
Board: /usr/local/lib/libopencv_core.so.4.1.1
Board: CMakeFiles/Board.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/hobie/Documents/Comp Vision/hobie/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Board"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Board.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Board.dir/build: Board

.PHONY : CMakeFiles/Board.dir/build

CMakeFiles/Board.dir/requires: CMakeFiles/Board.dir/board.cpp.o.requires

.PHONY : CMakeFiles/Board.dir/requires

CMakeFiles/Board.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Board.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Board.dir/clean

CMakeFiles/Board.dir/depend:
	cd "/home/hobie/Documents/Comp Vision/hobie" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/hobie/Documents/Comp Vision/hobie" "/home/hobie/Documents/Comp Vision/hobie" "/home/hobie/Documents/Comp Vision/hobie" "/home/hobie/Documents/Comp Vision/hobie" "/home/hobie/Documents/Comp Vision/hobie/CMakeFiles/Board.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Board.dir/depend
