BIN_NAME=solver

SRC_FILES=task1.cpp

CXX=g++
CXX_FLAGS = --std=c++11 -O3 -fopenmp

build:
	$(CXX) $(CXX_FLAGS) $(SRC_FILES) -o $(BIN_NAME)


run: build
	./$(BIN_NAME) 1000 1000 2 1 1
