# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/main.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/main.cpp.o -c /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/main.cpp

CMakeFiles/demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/main.cpp > CMakeFiles/demo.dir/main.cpp.i

CMakeFiles/demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/main.cpp -o CMakeFiles/demo.dir/main.cpp.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/main.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/main.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /usr/local/lib/libopencv_tracking.4.5.2.dylib
demo: /usr/local/lib/libopencv_highgui.4.5.2.dylib
demo: /usr/local/lib/libopencv_videoio.4.5.2.dylib
demo: /usr/local/lib/libopencv_plot.4.5.2.dylib
demo: /usr/local/lib/libopencv_datasets.4.5.2.dylib
demo: /usr/local/lib/libopencv_imgcodecs.4.5.2.dylib
demo: /usr/local/lib/libopencv_text.4.5.2.dylib
demo: /usr/local/lib/libopencv_ml.4.5.2.dylib
demo: /usr/local/lib/libopencv_video.4.5.2.dylib
demo: /usr/local/lib/libopencv_dnn.4.5.2.dylib
demo: /usr/local/lib/libopencv_calib3d.4.5.2.dylib
demo: /usr/local/lib/libopencv_features2d.4.5.2.dylib
demo: /usr/local/lib/libopencv_imgproc.4.5.2.dylib
demo: /usr/local/lib/libopencv_flann.4.5.2.dylib
demo: /usr/local/lib/libopencv_core.4.5.2.dylib
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo

.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build /Users/runge.liu/Documents/code/studyML/cpp/opencv_tracking_tld/build/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

