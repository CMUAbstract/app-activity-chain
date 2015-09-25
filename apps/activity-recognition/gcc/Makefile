EXEC = ar.out

OBJECTS = \
	main.o \

include ../../../Makefile.common.gcc

CFLAGS += \
	-I $(LIBCHAIN_ROOT)/src/include \

LFLAGS = \
	-L $(LIBCHAIN_ROOT)/gcc \

LIBS = \
    -lchain \

VPATH = $(SRC_DIR)
