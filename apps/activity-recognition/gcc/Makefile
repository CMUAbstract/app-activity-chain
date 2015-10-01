EXEC = ar.out

OBJECTS = \
	main.o \

include ../../../Makefile.common.gcc

CFLAGS += \
	-I $(LIBCHAIN_ROOT)/src/include \
	-I $(WISP_BASE_ROOT) \
	-I $(WISP_BASE_ROOT)/Sensors \
	-I $(WISP_BASE_ROOT)/wired \

LFLAGS = \
	-L $(LIBCHAIN_ROOT)/gcc \
	-L $(WISP_BASE_ROOT)/gcc \

LIBS = \
    -lchain \
	-lwisp-base \
	-lm \

VPATH = $(SRC_DIR)
