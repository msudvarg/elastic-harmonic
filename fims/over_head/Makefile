CXX = g++
CXXFLAGS = -Wall -std=c++17

LDFLAGS = 

SOURCES = overhead.cpp

OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = overhead

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)  # Tab before $(CXX)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<  # Tab before $(CXX)

# Clean rule
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)  # Tab before rm
