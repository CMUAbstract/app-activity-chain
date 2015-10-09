EXEC = ar.out

OBJECTS = \
	main.o \

include $(MAKER_ROOT)/Makefile.gcc

CFLAGS += \
	-I $(LIBCHAIN_ROOT)/src/include \
	-I $(WISP_BASE_ROOT) \
	-I $(WISP_BASE_ROOT)/Sensors \
	-I $(WISP_BASE_ROOT)/wired \

ifdef LIBCHAIN_ENABLE_DIAGNOSTICS
CFLAGS += -DLIBCHAIN_ENABLE_DIAGNOSTICS
endif

LFLAGS += \
	-L $(LIBCHAIN_ROOT)/bld/gcc \
	-L $(LIBMSPCONSOLE_ROOT)/bld/gcc \
	-L $(WISP_BASE_ROOT)/gcc \

LIBS = \
    -lchain \
	-lmspconsole \
	-lwisp-base \
	-lm \

VPATH = ../../src
