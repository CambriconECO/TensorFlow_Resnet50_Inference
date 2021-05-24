
CFLAGS=-I /usr/local/neuware/include/ -g  -std=c++11
CXXFLAGS=$(CFLAGS)

LDFALGS=-L /usr/local/neuware/lib64 `pkg-config --libs opencv` 

CXX=g++

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:%.cpp=%.o)

TARGET=cnrt_resnet50_demo

all:$(TARGET)

$(TARGET):$(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFALGS) -lcnrt -lcndrv -lpthread 
	
clean:
	rm  -f $(TARGET) $(OBJS)

